from recorder import record_audio
from time import sleep

# Replace <Your name> with your name from this list: [
# Omar
# Abdulrahman
# Abdullah
# Ahmed
# ]

# Replace <no 1 - 40> with the number of the record

# Don't forget change this number after each record

# Examples:
# record_audio(4, f"../<Your name>/open_middle_door/<Your name>_open{i}.wav")
# record_audio(4, f"../<Your name>/grant_me_access/<Your name>_grant{i}.wav")
# record_audio(4, f"../<Your name>/unlock_the_gate/<Your name>_unlock{i}.wav")

for i in range(1, 31):
    record_audio(2, f"../Omar/unlock_the_gate/Omar_unlock{i}.wav")
    sleep(3)
