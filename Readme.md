### Guide to run the repository...

1. ###### Preperation for Trainig process...
- Create a directory (dest_wav_file in my case), where all the audio .wav files will be stored. This .wav files will be used for training the model(Acoustic model).
- Additionaly I create 2 (train.json & test.json) json files that will contain a `key`:- "location of the .wav file" and `text`:- "text content of the .wav file".

The Preperation for training has been completed, now let's train the model...

2. Run `trainer.py` with required argument.
This will train the model for the given training data set and save the model checkpoints in the provided directory.

3. After training the model it's time to test it. I have created a python script named `engine.py` which is used for the inference. Run `engine.py` with the required parameters.
<u>Note</u> :- Provide a trained kenLM language model which is used for the decoding purpose.