import subprocess
from os import walk
import shutil
from tkinter import COMMAND

root_folder = "E:\\dataset malware CICD dataset 2020\\"
SOURCE_FOLDER = root_folder + "SMSDir\\SMS"
RUN_FOLDER = SOURCE_FOLDER + "\\run"
DONE_FOLDER = SOURCE_FOLDER + "\\done"
ERROR_FOLDER = SOURCE_FOLDER + "\\error"

CONTAINER_ID = "48fe7321e54b65f2aa39e5c5fbe863067705a14503ce9c0293216d01237c2ad7"
COMMAND = 'docker start -a -i ' + CONTAINER_ID

def run_docker(file_name):
    try:
        # Chạy phân tích 1 file apk
        subprocess.check_output(COMMAND, shell=True)
 
        # Move file đã phân tích thành công sang thư mục DONE_FOLDER
        old_path = RUN_FOLDER + "\\samples\\" + file_name
        new_path = DONE_FOLDER + "\\" + file_name
        shutil.move(old_path, new_path)
    except:
        # Move file phân lỗi vào rhuw mục ERROR_FOLDER
        old_path = RUN_FOLDER + "\\samples\\" + file_name
        new_path = ERROR_FOLDER + "\\" + file_name
        shutil.move(old_path, new_path)

sources = next(walk(SOURCE_FOLDER), (None, None, []))[2] #Lấy các files trong thư mục sources
for file_name in sources:
    print(file_name)
    # Move 1 file ở SOURCE_FOLDER sang RUN_FOLDER để phân tích
    old_path = SOURCE_FOLDER + "\\" + file_name
    new_path = RUN_FOLDER + "\\" + file_name
    shutil.move(old_path, new_path)
    #Chạy phân tích
    run_docker(file_name)