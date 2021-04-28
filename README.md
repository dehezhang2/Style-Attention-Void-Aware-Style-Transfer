# Final Year Project -- Style-Attention-Void-Aware Style Transfer
![Youtube Link](https://www.youtube.com/watch?v=3W4rLDDVyAQ)

A style-attention-void-aware style transfer model that learns the blank-leaving information during the style transfer.

![Screen Shot 2021-04-20 at 2.28.32 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%202.28.32%20PM-8903311.png)

## Overview

Arbitrary-Style-Per-Model fast neural style transfer has shown great potential in the academic field. Although state-of-the-art algorithms have great visual effect and efficiency, they are unable to address the blank-leaving (or void) information in specific artworks (e.g. traditional Chinese artworks). The available algorithms always try to maintain the similarity of details in the images before and after transformation, but certain details are often left blank in the artworks. 

This is my final year project, which aims to utilize the style attention map to learn the voidness information during the style transfer process. The main contributions of this project are a novel self-attention algorithm to extract the voidness information in the content and style image, and a novel style transfer module guided by the attention mask to swap the style. 

## Prerequisites

* Environment

  * Python (version 3.7.6)
  * CUDA 
    * CUDA version 10.0.130
    * CUDA Patch version 10.0.130.1
  * Algorithm
    * Pytorch (version 1.5.0)
    * torchvision
  * Training & Testing
    * Numpy
    * Matplotlib
    * Pillow
    * tqdm
    * streamlit (Graphic user interface)

  Anaconda environment recommended here!

  * GPU environment: NVIDIA GeForce GTX 1080 TI 

* Download the datasets
  * Content dataset: [MS-COCO](https://cocodataset.org/#home) is used to train the self-attention and SAVA-Net. 
  * Style dataset: [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) is used to train the SAVA-Net.

## Usage

### Test

1. Clone this repository

   ```shell
   git clone https://github.com/dehezhang2/Final_Year_Project.git
   cd Final_Year_Project
   ```

2. Prepare your content image and style image, and save the content image to `./testing_data/content` the style iamge to `./testing_data/style`. I also provide some in these two directories. 

3. Open the graphic user interface

   * Run the command line

     ```shell
     cd ./codes/transfer/
     streamlit run demo.py
     ```

   * Click the URL (or use forwarded ports)

![Screen Shot 2021-04-20 at 2.18.28 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%202.18.28%20PM-8899518.png)

4. Choose the content and style images

   ![Screen Shot 2021-04-20 at 2.19.35 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%202.19.35%20PM-8899589.png)

5. Click the `Start Transfer` button, and the attention maps, attention masks, and the relative frequency map of the content and style images will be visualised. The output will be shown. 

   ![Screen Shot 2021-04-20 at 2.20.35 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%202.20.35%20PM-8899708.png)

6. You can find the transfer output and attention maps in `Final_Year_Project/testing_data/result`.

### Train

1. Clone this repository

   ```shell
   git clone https://github.com/dehezhang2/Final_Year_Project.git
   cd Final_Year_Project
   ```

2. Download the training datasets, and change the file structure

   * All the content image should be in the directory `./training_data/content_set/val2014`
   * All the style image should be in the directory  `./training_data/style_set/val2014`

3. Filter the images by using two python files

   ```shell
   cd ./codes/data_preprocess/
   python filter.py
   python filter_percentage.py
   ```

4. We have two training phases:

   ![Screen Shot 2021-04-20 at 3.26.32 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%203.26.32%20PM-8903694.png)

   * Phase I training: train the self-attention module

   ```shell
   cd ./codes/transfer/
   python train_attn.py --dataset_dir ../../training_data/content_set/val2014
   ```

   * Phase II training: train the style transfer module

   ```shell
   python train_sava.py --content_dir ../../training_data/content_set/val2014 --style_dir ../../training_data/style_set/val2014 --save_dir ../../models/sava_training_hard
   ```

## Result

Here is a comparison of self-attention map used in [AAMS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Attention-Aware_Multi-Stroke_Style_Transfer_CVPR_2019_paper.pdf) (a) and our result (b)

![Screen Shot 2021-04-20 at 3.26.43 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%203.26.43%20PM-8903811.png)

Some results of content-style pairs are shown below (a) is our algorithm with attention masks, (b) is [SA-Net](https://arxiv.org/abs/1812.02342):

![Screen Shot 2021-04-20 at 1.17.45 PM](https://github.com/dehezhang2/Final_Year_Project/blob/master/README.assets/Screen%20Shot%202021-04-20%20at%201.15.53%20PM.png)



## Note

Although we have two contributions on the style transfer theory, there are limitations for this project:

* Principle of some settings cannot be well explained by theory.
  * Feature map projection method (ZCA for attention map, AdaIN for style transfer)
  * Method to train the self-attention module (similar to [AAMS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Attention-Aware_Multi-Stroke_Style_Transfer_CVPR_2019_paper.pdf))
* The limitation of computational resource.
  * The VGG decoder may not be properly trained.
  * It is diffcult to add attention loss to match the statistics of the style and output attention maps. 
  * It is difficult to divide the attention map into more clusters

## Acknowledgement

* I express gratitude to [AAMS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Attention-Aware_Multi-Stroke_Style_Transfer_CVPR_2019_paper.pdf) and [SA-Net](https://arxiv.org/abs/1812.02342), we benefit a lot from both their papers and codes. 
* Thanks to [Dr. Jing LIAO](https://liaojing.github.io/html/index.html). She has provided many insightful suggestions, such as the use of style attention, soft correlation mask, and attention loss to match the voidness statistics. I would like to express my sincere appreciation to [Kaiwen Xue](https://github.com/KevinRSX), who has provided many intelligent ideas on this project and helped me with part of the implementation. 

## Contact

If you have any questions or suggestions about this project, feel free to contact me by email<dehezhang2@gmail.com>.
