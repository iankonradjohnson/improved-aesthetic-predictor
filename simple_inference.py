# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

#####  This script will predict the aesthetic score for this image file:

img_path = "/Users/iankonradjohnson/Downloads/test-images/Screenshot 2025-05-03 at 11.14.27 PM.png"





# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

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

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

# Check for MPS (Apple Silicon), CUDA, or fall back to CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load the model with appropriate device mapping
s = torch.load("sac+logos+ava1-l14-linearMSE.pth", map_location=torch.device(device))   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to(device)
model.eval()
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   


pil_image = Image.open(img_path)

image = preprocess(pil_image).unsqueeze(0).to(device)



with torch.no_grad():
   image_features = model2.encode_image(image)

im_emb_arr = normalized(image_features.cpu().detach().numpy() )

tensor_input = torch.from_numpy(im_emb_arr).to(device)
# Use the appropriate tensor type based on device
if device == "cuda":
    tensor_input = tensor_input.type(torch.cuda.FloatTensor)
elif device == "mps":
    tensor_input = tensor_input.type(torch.FloatTensor).to(device)  # Explicitly move to MPS device
else:
    tensor_input = tensor_input.type(torch.FloatTensor)
    
prediction = model(tensor_input)

print( "Aesthetic score predicted by the model:")
print( prediction )


