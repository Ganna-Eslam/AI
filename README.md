Machine Learning

1- Data Analysis                                                                                                                                                                              
2- Visualization                                                                                                                                                                              
3- Data Preprocessing                                                                                                                                                                         
4- Encoding                                                                                                                                                                                   
4 - Scaling                                                                                                                                                                                   
5- UnBalanced Data Processing (using SMOTE)                                                                                                                                                   
6 - Machine Learning Models                                                                                                                                                                   
    1.Logistic Regression                                                                                                                                                                                                                                                                                                                                        
    2.Decision Tree                                                                                                                                                                               
    3.Random Forest                                                                                                                                                                                    
    4.SVM                                                                                                                                                                                               
    5.KNN                                                                                                                                                                                                
7- Deep Learning Models                                                                                                                                                                                       
8- Comparasion Between Models                                                                                                                                                                                  
8- Fine tuning (using RandomSearch)                                                                                                                                                                              


Deep Learning 

🔹 1. Multimodal Fusion for COVID-19 Classification
​Goal: To build a robust classification model for COVID-19 detection by integrating features from multiple data sources.
​Data: Multimodal dataset including CXR (Chest X-Ray) images, CT scans, and Cough Sound spectrograms.
​Methodology: Compared Early Fusion and Intermediate Fusion strategies. Applied Transfer Learning using pre-trained VGG16 and ResNet50 models as feature extractors.
​Best Results (Intermediate Fusion): Achieved a Validation Accuracy of 95.70% (0.95703).
​🔹 2. Arabic Sentiment Analysis (RNN, LSTM, Bi-LSTM)
​Goal: To classify the sentiment (Positive/Negative) of Arabic tweets.
​Data & Preprocessing: Used the Arabic Sentiment Twitter Corpus from Kaggle. Preprocessing involved comprehensive cleaning (URL/mention/hashtag/digit removal), punctuation stripping, and Arabic stop-words removal.
​Models: Implemented and compared three Recurrent Neural Network (RNN) architectures: SimpleRNN, LSTM, and Bidirectional LSTM (Bi-LSTM).
​Best Results:
​SimpleRNN Accuracy: 64.17% (0.64178)
​Bi-LSTM Accuracy (Best): 93.33% (0.9333)
​🔹 3. Abstractive Summarization using Seq2Seq with Attention
​Goal: To generate concise, abstractive summaries from source texts.
​Architecture: Developed a Sequence-to-Sequence (Seq2Seq) model using an Encoder-Decoder architecture.
​Models/Layers: Leveraged LSTM layers for both the Encoder (encoder_lstm) and Decoder (decoder_lstm). The model was enhanced with a custom Attention Mechanism to improve output quality.
​Metrics (ROUGE F1):
​ROUGE-1 F1: 0.4037
​ROUGE-2 F1: 0.2521
​ROUGE-L F1: 0.3703
​🔹 4. Generative Adversarial Network (GAN) for Abstract Art
​Goal: To train a model capable of generating novel, high-quality Abstract Art images.
​Data & Image Size: Used the Abstract Art Gallery dataset, with images processed to 28x28 pixels.
​Architecture: Standard GAN with coupled Generator and Discriminator networks.
​Generator: Used Conv2DTranspose layers to upscale the latent noise vector into an image.
​Discriminator: Used Conv2D layers with LeakyReLU and BatchNormalization to classify images as real or generated.


Computer Vision 


🔹 1. Brain Tumor Segmentation – LGG MRI (U-Net)
​Goal: Developed and trained an Image Segmentation model to accurately delineate tumor boundaries (masks) in Low-Grade Glioma (LGG) MRI scans.
​Methodology: Utilized the U-Net architecture (built with TensorFlow/Keras and Adamax optimizer) on the lgg-mri-segmentation dataset.
​Results: The model achieved strong performance on the test set:
​Dice Coefficient: 0.8959
​IoU Coefficient: 0.8249


​🔹 2. Traffic Sign Object Detection (YOLOv12)
​Goal: Trained a high-accuracy Object Detection model to identify traffic signs in various environments.
​Methodology: Employed the YOLOv12 architecture using the Ultralytics framework.
​Dataset: placas-transito (Traffic Signs dataset).
​Results: The model demonstrated high prediction efficiency:
​mAP50: 0.9161
​mAP50-95: 0.7441


​🔹 3. Static Hand Gesture Recognition Pipeline
​Goal: Developed an end-to-end pipeline for recognizing static hand gestures from video frames.
​Methodology:
​Feature Extraction: Used MediaPipe Hands to extract 3D landmarks for a 30-frame sequence.
​Classification: Trained a Logistic Regression model on the extracted feature vectors for real-time gesture classification.


​🔹 4. Real-Time Face Recognition System
​Goal: Implemented a complete, robust system for face detection, recognition, and tracking in live video streams (Real-Time).
​Key Components:
​High-Precision Detection: Achieved using the MTCNN network.
​Deep Embedding: Used OpenFace to generate unique 128D face embeddings for identity verification.
​Robust Tracking: Integrated ByteTrack for stable, multi-object tracking of faces within the video stream.


​🔹 5. Multi-Task NLP Pipeline (Hugging Face & Gemini API)
​Goal: Built an automated, multi-stage pipeline for processing and analyzing audio and text content, leveraging advanced pre-trained models.
​Models Used:
​Automatic Speech Recognition (ASR): Utilized the specialized MohamedRashad/Arabic-Whisper-CodeSwitching-Edition model for accurate Arabic transcription.
​Translation: Used the large facebook/nllb-200-distilled-600M model for high-quality translation (Arabic to English).
​Advanced Correction & Summarization: Integrated the Gemini API with the gemini-2.5-flash model to perform advanced text correction and prepare for summarization of the translated content.




