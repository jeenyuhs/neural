from torch import nn, Tensor, cuda

# CNN classifier.
class ImageClassifier(nn.Module):
	def __init__(self):
		super().__init__()

		# Den første konvolutionerende block
		self.conv_layer_1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2)
		)

		self.conv_layer_2 = nn.Sequential(
			nn.Conv2d(64, 512, 3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2)
		)

		self.conv_layer_3 = nn.Sequential(
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512),
			nn.MaxPool2d(2)
		) 

		self.classifier = nn.Sequential(
			nn.Flatten(start_dim=1),
			nn.Linear(in_features=512*64*64, out_features=2)
		)

	def forward(self, x: Tensor):
		x = self.conv_layer_1(x)
		x = self.conv_layer_2(x)
		x = self.conv_layer_3(x)
		x = self.classifier(x)
		return x

model = ImageClassifier().to("cuda" if cuda.is_available() else "cpu")