# if not running as script, change following variables manually
SEED = 5
BATCH_SIZE = 128
EPOCH_NUMBER = 5
testsize_per_channel = 800
png_header = "train_cnn"
plot_frequency = 5
nc_image_folders = ["/Users/sierra/Dropbox/datsets/nu_mu/data_bkp_8_1/QES/NC"]
cc_image_folders = ["/Users/sierra/Dropbox/datsets/nu_mu/data_bkp_8_1/QES/CC"]
model_name = "my_model.pt"


# uncomment the following if running as script
"""
import argparse
parser = argparse.ArgumentParser(description='Train a convolutional neural network for neutrino interactions. ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", type = int, default = 5, help = "Set random seed")
parser.add_argument("--cc_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES/CC1", nargs='*', help = "Name of folder containing CC interactions")
parser.add_argument("--nc_folder", type = str, default = "/home/sthoma31/neutrino_interaction_images/nu_mu_700/data/QES/NC1", nargs='*', help = "Name of folder containing NC interactions")
parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size when training")
parser.add_argument("--epoch", type = int, default = 20, help = "Number of epochs when training")
parser.add_argument("--testing_data_size", type = int, default = 100, help = "Size of testing data for each channel ")
parser.add_argument("--png_header", type = str, default = "trial", help = "Header name for PNG files")
parser.add_argument("--plot_freq", type = int, default = 5, help = "Plot confusion matrices every {plot_freq} times")
parser.add_argument("--model_name", type = str, default = "./recent.pt", help = "Name of model to save as")
args = parser.parse_args()
torch.manual_seed(args.seed)
BATCH_SIZE = args.batch_size
EPOCH_NUMBER = args.epoch
testsize_per_channel = args.testing_data_size
png_header = args.png_header
plot_frequency = args.plot_freq
nc_image_folders = args.nc_folder
cc_image_folders = args.cc_folder
model_name = args.model_name
"""
  
