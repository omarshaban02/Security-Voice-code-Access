import dejavu

# Load a Dejavu instance
djv = dejavu.Dejavu(config='./dejavu.conf')

# Load your audio file
fingerprint1 = djv.fingerprint_file("D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open1.wav")
fingerprint2 = djv.fingerprint_file("D:/CUFE/SBE/3rd/1st term/DSP/task5/test/recorder/open2.wav")

# Print the extracted fingerprints
print(fingerprint1)
print(fingerprint2)
