import torch
import torchaudio

class AduoUtil():
    """Pre-process the audio data and make it audio of equal duaration by trucation and padding"""
    @staticmethod
    def openAudioFile(audioFilePath):
        waveform, sample_rate = torchaudio.load(audioFilePath)

        return waveform, sample_rate
    
    @staticmethod
    def trunc(wave, sample_len = 480000):

        if(wave.shape[1] > sample_len):
            wave = wave[:,:sample_len]
        
        return wave

    @staticmethod
    def padder(wave, sample_len = 480000):
        audio_len = wave.shape[1]
        if(audio_len < sample_len):
            num_missing_samples = sample_len - audio_len
            adder = (0, num_missing_samples)
            wave = torch.nn.functional.pad(wave,adder)
        
        return wave


if __name__ == "__main__":
    audiofilepath = 'dest_wav_file\84-121123-0000.wav'


    audioProcessor = AduoUtil
    wave , sample_rate = audioProcessor.openAudioFile(audiofilepath)
    print("initial shape : ",wave.shape)

    wave =audioProcessor.trunc(wave)
    print("Trucated shape : ",wave.shape)

    wave = audioProcessor.padder(wave)
    print("padded shape : ",wave.shape)