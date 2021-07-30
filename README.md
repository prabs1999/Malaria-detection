# Malaria-detection

The Malaria Detection from thin film blood smear images demands segmentation of single blood cells from the microscopic blood slide images which can be taken from a pathologist and the dataset would contain cell images that are not segmented. Hence, segmentation in the proposed method is done using a variety of image processing techniques.

Edge detection techniques and segmentation techniques used in this system overcomes the issue of overlapping of cells by eliminating the noise and finding the discontinuities of the cells. It differentiates each cell and detects the infection in the cell using morphological segmentation.

Also, all the images are raw and have different intensities, and since there is no uniformity in all the images, detection of cells and infection is very difficult. To overcome this problem, the proposed method uses histogram matching where all the images are standard and has the same intensity which in turn increases the accuracy level

dataset: 1)	https://www.kaggle.com/search?q=malaria+detection
         2) ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip
         
