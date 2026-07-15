import torch
import torch.nn as nn
from torch.autograd import Function

# Falsification corresponding to EEG Dataset (Chen 2024 / raw EEG)
# Failed due to: MATLAB Preprocessing Corruption & Domain Shortcuts
# DANN representation collapse where classifiers memorized subject-specific spatial shortcuts 
# rather than extracting cognitive state signals.

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN_Classifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=2, num_domains=10):
        super(DANN_Classifier, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cognitive State Classifier (Intent)
        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Domain Classifier (Subject ID)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )

    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict cognitive state
        class_output = self.class_classifier(features)
        
        # Reverse gradients for domain adaptation
        reverse_features = ReverseLayerF.apply(features, alpha)
        
        # Predict subject (Domain)
        domain_output = self.domain_classifier(reverse_features)
        
        # Falsification result: Domain shortcuts bypassed the DANN mechanism.
        # The model memorized the spatial shortcuts (MATLAB preprocessing artifacts)
        # resulting in representation collapse.
        
        return class_output, domain_output
