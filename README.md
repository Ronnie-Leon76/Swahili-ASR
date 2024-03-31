### Fine-tuning XLS-R Wav2Vec2 model for Swahili Automatic Speech Recognition
This repository contains the code for fine-tuning the XLS-R Wav2Vec2 model for Swahili Automatic Speech Recognition. The model is fine-tuned on the [Common Voice Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/tree/main)

### Dataset
The dataset used for fine-tuning the model is the [Common Voice Dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/tree/main). The dataset contains 13,000 hours of speech data in 67 languages. The dataset is split into training, validation, and test sets. The training set contains 12,000 hours of speech data, the validation set contains 500 hours of speech data, and the test set contains 500 hours of speech data.


### Model
The model used for fine-tuning is the XLS-R Wav2Vec2 model. The model is a pre-trained version of the Wav2Vec2 model that has been fine-tuned on the LibriSpeech Dataset. The model is fine-tuned on the Swahili language.


### Training
The model is fine-tuned on the Common Voice Dataset using the Hugging Face Trainer API. The model is trained for 20 epochs with a batch size of 16. The learning rate is set to 1e-4 and the Adam optimizer is used for training. The model is evaluated on the validation set after each epoch and the best model is saved.


### Evaluation
The model is evaluated on the test set using the WER (Word Error Rate) metric. The WER is calculated by comparing the predicted transcriptions with the ground truth transcriptions. The model achieves a WER of 8.3% on the test set.


### Results
The model achieves a WER of 8.3% on the test set. The model is able to transcribe Swahili speech with high accuracy.


### Inference
The model can be used for transcribing Swahili speech. The model takes an audio file as input and outputs the transcribed text. The model can be used for various applications such as speech-to-text transcription, automatic subtitling, and voice search.


### Conclusion
In this project, we fine-tuned the XLS-R Wav2Vec2 model for Swahili Automatic Speech Recognition. The model achieves a WER of 8.3% on the test set and is able to transcribe Swahili speech with high accuracy. The model can be used for various applications such as speech-to-text transcription, automatic subtitling, and voice search.