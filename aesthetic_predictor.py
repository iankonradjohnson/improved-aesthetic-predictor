import clip
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class AestheticPredictor:
    def __init__(self, model_path="sac+logos+ava1-l14-linearMSE.pth", clip_model="ViT-L/14", device=None):
        """
        Initialize the aesthetic predictor with the specified model.
        
        Args:
            model_path (str): Path to the aesthetic model weights
            clip_model (str): CLIP model to use (default: ViT-L/14)
            device (str, optional): Device to run on. If None, will auto-detect.
        """
        # Determine device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Initialize MLP model
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        
        # Check if model path is absolute or relative
        if not os.path.isabs(model_path):
            # If relative, use the current file's directory as base
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_path)
            
        # Load model weights with appropriate device mapping
        model_weights = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
    
    def _normalized(self, a, axis=-1, order=2):
        """Normalize the input array."""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)
    
    def predict(self, image):
        """
        Predict the aesthetic score for a given image.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            float: Aesthetic score (between 1-10)
        """
        # Handle string input (image path)
        if isinstance(image, str):
            image = Image.open(image)
        # Ensure input is a PIL Image
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image or a path to an image file")
        
        # Preprocess the image for CLIP
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image features from CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_image)
        
        # Normalize features
        im_emb_arr = self._normalized(image_features.cpu().detach().numpy())
        
        # Prepare tensor input with appropriate type
        tensor_input = torch.from_numpy(im_emb_arr).to(self.device)
        if self.device == "cuda":
            tensor_input = tensor_input.type(torch.cuda.FloatTensor)
        elif self.device == "mps":
            tensor_input = tensor_input.type(torch.FloatTensor).to(self.device)
        else:
            tensor_input = tensor_input.type(torch.FloatTensor)
        
        # Get prediction from model
        prediction = self.model(tensor_input)
        
        # Return prediction as a float
        return float(prediction.item())

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = AestheticPredictor()
    
    # Example with image path
    img_path = "test.jpg"  # replace with your image path
    if os.path.exists(img_path):
        score = predictor.predict(img_path)
        print(f"Aesthetic score for {img_path}: {score:.2f}/10")
    
    # You could also use a PIL Image directly:
    # from PIL import Image
    # image = Image.open("test.jpg")
    # score = predictor.predict(image)
    # print(f"Aesthetic score: {score:.2f}/10")