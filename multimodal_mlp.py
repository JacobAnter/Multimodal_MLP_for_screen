import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
import json


class nnPULoss(nn.Module):
    def __init__(self, prior, beta=0.0, gamma=1.0):
        """
        prior: class prior P(y=1)
        beta: non-negative correction threshold (usually 0)
        gamma: scaling for negative risk
        """
        super().__init__()
        self.prior = prior
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        logits: (B,)
        labels: (B,) with 1 (positive) or 0 (unlabeled)
        """

        probs = torch.sigmoid(logits)

        pos_mask = labels == 1
        unl_mask = labels == 0

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        # positive risk
        pos_loss = F.binary_cross_entropy(probs[pos_mask], torch.ones_like(probs[pos_mask]), reduction='mean')

        # negative risk (from unlabeled)
        neg_loss = F.binary_cross_entropy(probs[unl_mask], torch.zeros_like(probs[unl_mask]), reduction='mean')

        # correction term
        neg_risk = neg_loss - self.prior * F.binary_cross_entropy(
            probs[pos_mask], torch.zeros_like(probs[pos_mask]), reduction='mean'
        )

        # nnPU correction
        if neg_risk < -self.beta:
            neg_risk = -self.gamma * neg_risk

        risk = self.prior * pos_loss + neg_risk
        return risk


class MultiModalPUPredictor(pl.LightningModule):
    def __init__(
        self,
        pheno_dim=3,
        ppi_dim=440,
        hidden_pheno=16,
        hidden_ppi=64,
        fusion_dim=64,
        lr=1e-3,
        prior=0.1,
        l1_lambda=0.0,
        use_ppi=True,
        fusion_type='concat', # 'concat', 'attention', 'matmul'
        ppi_net_type='mlp'    # 'mlp', 'attention'
    ):
        super().__init__()

        self.save_hyperparameters()
        self.use_ppi = use_ppi
        self.fusion_type = fusion_type
        self.ppi_net_type = ppi_net_type

        # ------------------
        # Phenotype branch
        # ------------------
        self.pheno_net = nn.Sequential(
            nn.Linear(pheno_dim, hidden_pheno),
            nn.LayerNorm(hidden_pheno),
            nn.ReLU(),
            nn.Dropout(0.1),  # mild

            nn.Linear(hidden_pheno, hidden_pheno),
            nn.LayerNorm(hidden_pheno),
            nn.ReLU()
        )

        # ------------------
        # PPI branch
        # ------------------
        if self.use_ppi:
            if self.ppi_net_type == 'mlp':
                self.ppi_net = nn.Sequential(
                    nn.Linear(ppi_dim, hidden_ppi),
                    nn.LayerNorm(hidden_ppi),
                    nn.ReLU(),
                    nn.Dropout(0.3),  # stronger regularization

                    nn.Linear(hidden_ppi, hidden_ppi),
                    nn.LayerNorm(hidden_ppi),
                    nn.ReLU()
                )
            elif self.ppi_net_type == 'attention':
                # Replace MLP with a self-attention mechanism
                # Project the input vector, treat it as a sequence of length 1, apply attention
                self.ppi_proj = nn.Linear(ppi_dim, hidden_ppi)
                self.ppi_attn = nn.MultiheadAttention(embed_dim=hidden_ppi, num_heads=4, batch_first=True, dropout=0.3)
                self.ppi_norm = nn.LayerNorm(hidden_ppi)
                self.ppi_ff = nn.Sequential(
                    nn.Linear(hidden_ppi, hidden_ppi),
                    nn.ReLU()
                )
            else:
                raise ValueError(f"Unknown ppi_net_type: {self.ppi_net_type}")

        # ------------------
        # Fusion
        # ------------------
        if not self.use_ppi:
            # Ablation study: Only Phenotype features used
            self.fusion_net = nn.Sequential(
                nn.Linear(hidden_pheno, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1)
            )
        else:
            if self.fusion_type == 'concat':
                input_dim = hidden_pheno + hidden_ppi
            elif self.fusion_type == 'matmul':
                # Performs a matrix multiplication of the embeddings:
                # (B, hidden_ppi, 1) @ (B, 1, hidden_pheno) -> (B, hidden_ppi, hidden_pheno)
                # Then flattened. This conceptually matches the nx1 @ 1xm dimensional transformation.
                input_dim = hidden_ppi * hidden_pheno
            elif self.fusion_type == 'attention':
                self.attn_dim = fusion_dim
                self.pheno_proj = nn.Linear(hidden_pheno, self.attn_dim)
                self.ppi_proj_attn = nn.Linear(hidden_ppi, self.attn_dim)
                self.cross_attn = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=4, batch_first=True)
                input_dim = self.attn_dim * 2
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")

            self.fusion_net = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Linear(fusion_dim, 1)
            )

        self.criterion = nnPULoss(prior=prior)
        self.lr = lr
        self.l1_lambda = l1_lambda

    def forward(self, pheno, ppi=None, return_embeddings=False):
        pheno_emb = self.pheno_net(pheno)
        embeddings_dict = {"pheno_emb": pheno_emb}

        if not self.use_ppi:
            fusion_emb = self.fusion_net[:-1](pheno_emb)
            logits = self.fusion_net[-1](fusion_emb).squeeze(-1)
            embeddings_dict["fusion_emb"] = fusion_emb
            if return_embeddings:
                return logits, embeddings_dict
            return logits

        # PPI forward
        if self.ppi_net_type == 'mlp':
            ppi_emb = self.ppi_net(ppi)
        elif self.ppi_net_type == 'attention':
            x = self.ppi_proj(ppi).unsqueeze(1) # (B, 1, hidden_ppi)
            attn_out, _ = self.ppi_attn(x, x, x)
            x = self.ppi_norm(x + attn_out)
            x = x + self.ppi_ff(x)
            ppi_emb = x.squeeze(1) # (B, hidden_ppi)
        
        embeddings_dict["ppi_emb"] = ppi_emb

        # Fusion
        if self.fusion_type == 'concat':
            fusion_input = torch.cat([pheno_emb, ppi_emb], dim=1)
        elif self.fusion_type == 'matmul':
            # Matrix multiplication between extracted features to fuse them
            # (B, hidden_ppi, 1) @ (B, 1, hidden_pheno) -> (B, hidden_ppi, hidden_pheno)
            batch_size = pheno_emb.size(0)
            res = torch.bmm(ppi_emb.unsqueeze(2), pheno_emb.unsqueeze(1))
            fusion_input = res.view(batch_size, -1)
        elif self.fusion_type == 'attention':
            # Attention-based fusion (Self-attention over projected embeddings)
            p_proj = self.pheno_proj(pheno_emb).unsqueeze(1) # (B, 1, attn_dim)
            p_ppi = self.ppi_proj_attn(ppi_emb).unsqueeze(1) # (B, 1, attn_dim)
            
            seq = torch.cat([p_proj, p_ppi], dim=1) # (B, 2, attn_dim)
            attn_out, _ = self.cross_attn(seq, seq, seq)
            
            fusion_input = attn_out.reshape(attn_out.size(0), -1) # (B, 2 * attn_dim)

        fusion_emb = self.fusion_net[:-1](fusion_input)
        logits = self.fusion_net[-1](fusion_emb).squeeze(-1)
        embeddings_dict["fusion_emb"] = fusion_emb

        if return_embeddings:
            return logits, embeddings_dict

        return logits

    def training_step(self, batch, batch_idx):
        pheno = batch["pheno"]
        ppi = batch.get("ppi") if self.use_ppi else None
        labels = batch["label"].float()

        logits, embeddings = self(pheno, ppi, return_embeddings=True)
        loss = self.criterion(logits, labels)

        if self.l1_lambda > 0:
            l1_loss = embeddings["pheno_emb"].abs().mean() + embeddings["fusion_emb"].abs().mean()
            if self.use_ppi:
                l1_loss += embeddings["ppi_emb"].abs().mean()
            loss = loss + self.l1_lambda * l1_loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pheno = batch["pheno"]
        ppi = batch.get("ppi") if self.use_ppi else None
        labels = batch["label"].float()

        logits = self(pheno, ppi)
        loss = self.criterion(logits, labels)

        probs = torch.sigmoid(logits)

        self.log("val_loss", loss)
        self.log("val_mean_prob", probs.mean())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser(description="Multimodal MLP execution script with flexible configurations")
    parser.add_argument('--config', type=str, help='Path to an optional JSON configuration file')
    
    # Core architectural arguments
    parser.add_argument('--use_ppi', type=lambda x: str(x).lower() == 'true', default=True, help='Use both modalities. If False, only pheno is used (ablation study).')
    parser.add_argument('--fusion_type', type=str, choices=['concat', 'matmul', 'attention'], default='concat', help='Fusion technique for the two modalities')
    parser.add_argument('--ppi_net_type', type=str, choices=['mlp', 'attention'], default='mlp', help='Architecture type for the PPI branch')
    
    # Dimensionality and hyperparameters
    parser.add_argument('--pheno_dim', type=int, default=3)
    parser.add_argument('--ppi_dim', type=int, default=440)
    parser.add_argument('--hidden_pheno', type=int, default=16)
    parser.add_argument('--hidden_ppi', type=int, default=64)
    parser.add_argument('--fusion_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Note: parse_known_args used to allow easy usage in PyTorch Lightning without breaking on unknown flags
    args, unknown = parser.parse_known_args()

    # Load config from JSON if provided, overriding defaults
    config = vars(args)
    if args.config:
        try:
            with open(args.config, 'r') as f:
                json_config = json.load(f)
                config.update(json_config)
        except Exception as e:
            print(f"Warning - failed to load config file {args.config}: {e}")

    print("\n" + "="*50)
    print("Running with configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")

    # Initialize model using the merged configuration
    model = MultiModalPUPredictor(
        pheno_dim=config.get('pheno_dim', 3),
        ppi_dim=config.get('ppi_dim', 440),
        hidden_pheno=config.get('hidden_pheno', 16),
        hidden_ppi=config.get('hidden_ppi', 64),
        fusion_dim=config.get('fusion_dim', 64),
        lr=config.get('lr', 1e-3),
        prior=config.get('prior', 0.1),
        l1_lambda=config.get('l1_lambda', 0.0),
        use_ppi=config.get('use_ppi', True),
        fusion_type=config.get('fusion_type', 'concat'),
        ppi_net_type=config.get('ppi_net_type', 'mlp')
    )

    print("Model initialized successfully!")


if __name__ == '__main__':
    main()