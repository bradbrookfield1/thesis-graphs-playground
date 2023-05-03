import wave

def main():
    concatenate_audio_wave(['Noise_1_2.wav', 'Noise_2_3.wav', 'Noise_3_4.wav', 'Noise_4_5.wav', 'Noise_5_6.wav', 'Noise_6_7.wav', 'Noise_7_8.wav', 'Noise_8_9.wav', 'Noise_9_10.wav'], 'noise_est.wav')

def concatenate_audio_wave(audio_clip_paths, output_path):
    data = []
    for clip in audio_clip_paths:
        w = wave.open(clip, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

if __name__ == '__main__':
    main()