import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import *

import os
from tkinter import messagebox
import hashlib

from main import detectLabel
from main import createRawCSV
from main import get_file
from main import readJson
from tkinter import Tk, filedialog
from main import scanFolder
from PIL import ImageTk, Image

from XuLy import *

from trichXuat import QuetJSON
from trichXuat import getFullinformation
#from data_Malware.action import creBtnasd

from CNN import fitVaoCNN

from ScanCNN import DetectMeOniiChan
from ScanCNN import QuetTrenFolder

import win32gui, win32con

from label import *

hide = win32gui.GetForegroundWindow()
win32gui.ShowWindow(hide , win32con.SW_HIDE)

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
hahahoho = ""


def BtnQuetClick():
    folder = filedialog.askdirectory()
    if(folder != ""):
        if os.path.isdir(folder):
            label_ThongBaoMoThuMuc.configure(text = "Mở Folder thành công")
            label_ThongBaoMoThuMuc.place(x = 100, y = 230)

            listFile = Listbox(app, height=19, width = 90)
            
            i = 1
            raw = ""
            for filename in os.listdir(folder):
                if(os.path.isdir(os.path.join(folder, filename))):
                    listFile.insert(i, "Folder: " + str(filename))
                    raw += "Folder " + str(filename) + "\n"
                else:
                    listFile.insert(i, "File: " + str(filename))
                    raw += "File " + str(filename) + "\n"


            global hahahoho
            hahahoho = raw


            answer = messagebox.askyesno("", "Bạn có muốn bắt đầu phân tích và lưu thông tin của file trong folder " + folder + " ?" )

            if answer:
                if("data.csv" not in os.listdir(os.getcwd())):
                    answer = messagebox.askyesno("", "Không tìm được file data.csv để lưu dữ liệu, tạo file mới?")
                    if answer:
                        createRawCSV()
                    else:
                        return

                for file_name in os.listdir(folder):
                    if os.path.isdir(os.path.join(folder, file_name)):
                        scanFolder(os.path.join(folder, file_name))
                    else:
                        get_file(os.path.join(folder, file_name))

                    state = "Trạng thái: Đã scan xong " + folder
                    label_trangThai.configure(text = state)

        else:
            label_ThongBaoMoThuMuc.configure(text = "Folder Không tồn tại")


checkValue = []

f = open("checkValue.option", "r")
for line in f:
    line = line.strip()
    checkValue.append(int(line))
f.close()


mydict = {
        "0":"safe",
        "1":"safe",
        "2":"error",
        "3":"trojan",
        "4":"virus",
        "5":"worm"
}

#print(checkValue)

checkValueDefault = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
Backup = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0]

ListDacTrung = [
    "FileName", 
    "MD5",
    "SHA-1",
    "SHA-256",
    "SHA-512",
    "e_magic",
    "e_cblp",
    "e_cp",
    "e_crlc",
    "e_cparhdr",
    "e_minalloc",
    "e_maxalloc",
    "e_ss",
    "e_sp",
    "c_csum",
    "e_ip",
    "c_cs",
    "e_lfarlc",
    "e_ovno",
    "e_oemid",
    "e_oeminfo",
    "e_lfanew",
    "Machine",
    "SizeOfOptionalHeader",
    "Characteristics",
    "Signature",
    "Magic",
    "MajorLinkerVersion",
    "MinorLinkerVersion",
    "SizeOfCode",
    "SizeOfInitializedData",
    "SizeOfUninitializedData",
    "AddressOfEntryPoint",
    "BaseOfCode",
    "BaseOfData",
    "ImageBase",
    "SectionAlignment",
    "FileAlignment",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "MajorImageVersion",
    "MinorImageVersion",
    "MajorSubsystemVersion",
    "MinorSubsystemVersion",
    "Reserved1",
    "SizeOfImage",
    "SizeOfHeaders",
    "CheckSum",
    "Subsystem",
    "DllCharacteristics",
    "SizeOfStackReserve",
    "SizeOfStackCommit",
    "SizeOfHeapReserve",
    "SizeOfHeapCommit",
    "LoaderFlags",
    "NumberOfRvaAndSizes",
    "LengthOfPeSections",
    "MeanEntropy",
    "MinEntropy",
    "MaxEntropy",
    "MeanRawSize",
    "MinRawSize",
    "MaxRawSize",
    "MeanVirtualSize",
    "MinVirtualSize",
    "MaxVirtualSize",
    "ImportsNbDLL",
    "ImportsNb",
    "ImportsNbOrdinal",
    "ExportNb",
    "ResourcesNb",
    "ResourcesMeanEntropy",
    "ResourcesMinEntropy",
    "ResourcesMaxEntropy",
    "ResourcesMeanSize",
    "ResourcesMinSize",
    "ResourcesMaxSize",
    "LoadConfigurationSize",
    "VersionInformationSize",
    "DLL",
    "LengthOfInformation",
]



def OpenBangDacTrung():
    newWindow = Toplevel(app)
    newWindow.title("List các đặc trưng sẽ lấy")
    newWindow.geometry("800x500")
    newWindow.iconbitmap("icon/Design.ico")
    '''
    CheckVar_extra_tree = IntVar()
    C2 = Checkbutton(app, text = "EXTRA TREE", variable = CheckVar_extra_tree, \
                 onvalue = 1, offvalue = 0, height=5, \
                 width = 20)
    C2.place(x =300, y=400)
    '''
    checkVar = []
    TickBtn = []


    main_frame = Frame(newWindow, width = 800, height = 500)
    main_frame.place(x=0,y=0)


    my_canvas = Canvas(main_frame, width = 800, height = 500)
    my_canvas.place(x=0,y=0)
    
    my_scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview) 
    my_scrollbar.place( x = 780, y = 0, height = 500)


    my_canvas.configure(yscrollcommand = my_scrollbar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

    def _on_mouse_wheel(event):
        try:
            my_canvas.yview_scroll(-1 * int((event.delta / 120)), "units")
        except:
            pass
    my_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)

    second_frame = Frame(my_canvas,width=800,height=500)
    second_frame.place(x=0,y=0)
    my_canvas.create_window((0,0), window=second_frame, anchor="nw")




    row = 30
    row2 = 1
    for i in range(len(ListDacTrung)):
        checkVar.append(0)
        TickBtn.append(0)

    def updateSoDacTrung():
        count = 0
        for i in checkVar:
            if(i.get() == 1):
                count += 1
        state = "Số đặc trưng: " + str(count) + "/81"
        NumberOfDacTrung = Label(second_frame, text = state)
        NumberOfDacTrung.place(x = 300, y = 150)


    for i in range(len(ListDacTrung)):
        checkVar[i] = IntVar()
        TickBtn[i] = Checkbutton(second_frame, text = str(i) + ". " +ListDacTrung[i], variable = checkVar[i], onvalue = 1, offvalue = 0, command = updateSoDacTrung)

        TickBtn[i].place(x = 100, y = i * 50)
    second_frame.configure(height = (i) * 50 + 70)

    for i in range(len(checkValue)):
        if(checkValue[i] == 1):
            checkVar[i].set(1)
        else:
            checkVar[i].set(0)



    def BtnOk():
        for i in range(len(checkVar)):
            if checkVar[i].get() == (1):
                checkValue[i] = 1
            else:
                checkValue[i] = 0

        f = open("checkValue.option", "w")
        for i in checkValue:
            f.write(str(i))
            f.write("\n")
        f.close()

        main_frame.destroy()
        my_canvas.destroy()
        second_frame.destroy()
        my_scrollbar.destroy()  
        newWindow.destroy()
        trichChon()
        
    
    ok = Button(second_frame, text = "OK!", padx = 30, pady = 10, command = BtnOk )
    ok.place(x = 300, y = 50)


    def BtnDeFault():
        for i in range(len(checkValueDefault)):
            if(checkValueDefault[i] == 1):
                checkVar[i].set(1)
            else:
                checkVar[i].set(0)
                count = 0
        for i in checkVar:
            if(i.get() == 1):
                count += 1
        state = "Số đặc trưng: " + str(count) + "/81"
        NumberOfDacTrung = Label(second_frame, text = state)
        NumberOfDacTrung.place(x = 300, y = 150)

    defaultBtn = Button(second_frame, text = "Mặc định", padx = 40, pady = 10, command = BtnDeFault)
    defaultBtn.place(x = 300, y = 100)

    count = 0
    for i in checkValue:
        if(i == 1):
            count += 1
    state = "Số đặc trưng: " + str(count) + "/81"
    NumberOfDacTrung = Label(second_frame, text = state)
    NumberOfDacTrung.place(x = 300, y = 150)

def trichChon():
    messagebox.showinfo("OK!", "Chọn file JSON để lấy đặc trưng")
    file = filedialog.askopenfilename(title='select', filetypes=(("Json files", "*.json"),)  )

    readJson(file, checkValue)
    messagebox.showinfo("OK!", "Done")



def trichXuat():
    messagebox.showinfo("!!!", "Hãy chọn folder trích xuất ra JSON")
    folder = filedialog.askdirectory()
    if(folder != ""):
        nameList = []
        QuetJSON(folder, nameList)

        listFile = Listbox(app, height=19, width = 90)
        
        i = 1
        raw = ""
        for filename in nameList:
            listFile.insert(i, "File: " + str(filename))

        raw += "File: " + str(filename) + "\n"
        global hahahoho
        hahahoho = raw

        messagebox.showinfo("!!!", "Done!")


def openTrichWindow():
    OpenBangDacTrung()

def TaoClassifier():
    if ("data.csv" in (os.listdir(os.getcwd()))):
        
        answer = messagebox.askyesno("", "Bạn có muốn đưa vào mô hình học máy không?")
        if answer:
            if("classifier.pkl" in os.listdir("classifier") and "features.pkl" in os.listdir("classifier")):
                answer = messagebox.askyesno("ops!!!", "Đã tìm thấy classifier rồi, bạn có muốn xóa cái cũ đi không?")
                if answer:
                    handleClassifier()
            else:
                handleClassifier()

        answer = messagebox.askyesno("", "Bạn có muốn đưa vào mô hình học sâu không?")

        if answer:
            if("CheckMate_without_callback.h5" in  (os.listdir(os.getcwd()))):
                answer = messagebox.askyesno("", "Đã có classifier rồi, bạn có muốn xóa cái cũ đi không?")
                if(answer):
                    messagebox.showinfo("OK!!!", "Giờ máy sẽ bắt đầu học theo mô hình học sâu CNN, bạn có thể quan sát bảng terminal nếu muốn biết quá trình.")
                    fitVaoCNN()
                    messagebox.showinfo("OK!!!", "Xong rồi!")
            else:
                messagebox.showinfo("OK!!!", "Giờ máy sẽ bắt đầu học theo mô hình học sâu CNN, bạn có thể quan sát bảng terminal nếu muốn biết quá trình.")
                fitVaoCNN()
                messagebox.showinfo("OK!!!", "Xong rồi!")
    else:
        messagebox.showinfo("ERROR!!!", "Không tìm thấy file file data.csv, hãy quét thư mục để xuất file trước!")

def QuetVirus():
    chon = 0
    filename = ""
    answer = messagebox.askyesno("AYY", "Bạn có muốn quét bằng Deep Learning không?")
    if(answer):
        if("CheckMate_without_callback.h5" not in  (os.listdir(os.getcwd()))):
            messagebox.showinfo("OMG!!!", "Không tìm thấy file h5 rồi, bạn phải đưa dữ liệu vào mô hình học sâu trước!")
        else:
            filename = filedialog.askopenfilename()
            chon = 1
            label = DetectMeOniiChan(filename, checkValue)
            messagebox.showinfo("OK", str(label))

    answer = messagebox.askyesno("AYY", "Bạn có muốn quét bằng Machine Learning không?")
    if answer:
        if("classifier.pkl" not in os.listdir("classifier") and "features.pkl" not in os.listdir("classifier")):
            messagebox.showinfo("OMG!!!", "Không tìm thấy file classifier rồi, bạn phải đưa dữ liệu vào mô hình học máy trước!")
        else:
            if(chon == 1):
                label = ScanVirusML(filename)
                messagebox.showinfo("OK", mydict[str(label)])
            else:
                filename = filedialog.askopenfilename()
                chon = 1
                label = ScanVirusML(filename)
                messagebox.showinfo("OK", mydict[str(label)])



def QuetVirusFolder():
    chon = 0
    answer = messagebox.askyesno("AYY", "Bạn có muốn quét bằng Deep Learning không?")
    if(answer):
        if("CheckMate_without_callback.h5" not in  (os.listdir(os.getcwd()))):
            messagebox.showinfo("OMG!!!", "Không tìm thấy file h5 rồi, bạn phải đưa dữ liệu vào mô hình học sâu trước!")
        else:
            messagebox.showinfo("!!!", "Hãy chọn folder để quét")
            chon = 1
            folder = filedialog.askdirectory()
            if folder != "":
                response = []
                nameList = []
                result = QuetTrenFolder(folder, nameList, response, checkValue)
                listFile = Listbox(app, height=19, width = 90)
                
                i = 1
                raw = ""
                for res in range(len(response)):
                    listFile.insert(i, "File: " + str(nameList[res]) + " là " + str(response[res]))
                    raw += str(i) + ". File: " + str(nameList[res]) + " là "+ str(response[res]) + "\n"
                    i += 1
                global hahahoho
                hahahoho = raw
                #print("\n\n============================\n" + hahahoho + "\n\n============================\n")

            tong_so_file_safe_thuc_te = 0
            for i in range(len(nameList)):
                if("Virus" not in nameList[i]):
                    tong_so_file_safe_thuc_te += 1

            tong_so_file = len(nameList)
            tong_so_file_malware_thuc_te = tong_so_file - tong_so_file_safe_thuc_te

            tong_so_file_safe_tim_duoc = 0
            tong_so_file_malware_tim_duoc = 0

            for i in range(len(response)):
                if("safe" in response[i]):
                    tong_so_file_safe_tim_duoc += 1

            tong_so_file_malware_tim_duoc = tong_so_file - tong_so_file_safe_tim_duoc

            wrong = 0
            accurancy = 100
            for i in range(len(nameList)):
                if ("Virus" not in nameList[i] and "safe" not in response[i]): # la safe ma danh sang virus
                    accurancy -= (1/tong_so_file) * 100
                    wrong += 1
                elif ("Virus" in nameList[i] and "safe" in response[i]): #la virus ma danh sang safe
                    accurancy -= (1/tong_so_file) * 100
                    wrong += 1

            statusBox = "DEEPLEARNING"
            statusBox += "\nĐã quét: " + str(tong_so_file) + "\nSố file an toàn tìm được: " + str(tong_so_file_safe_tim_duoc) + "\nSố file Malware tìm được: " + str(tong_so_file_malware_tim_duoc)
            statusBox += "\nSố file an toàn thực tế: " + str(tong_so_file_safe_thuc_te) + "\nSố file Malware thực tế: " + str(tong_so_file_malware_thuc_te)
            statusBox += "\nSố file tìm sai: " + str(wrong)
            statusBox += "\nĐộ chính xác: "

         


            statusBox += str(accurancy) + "%"
            messagebox.showinfo("Kết quả", statusBox)


    answer = messagebox.askyesno("AYY", "Bạn có muốn quét bằng Machine Learning không?")
    if answer:
        if("classifier.pkl" not in os.listdir("classifier") and "features.pkl" not in os.listdir("classifier")):
            messagebox.showinfo("OMG!!!", "Không tìm thấy classifier rồi, bạn phải đưa dữ liệu vào mô hình học máy trước!")
        else:
            if (chon == 0):
                folder = filedialog.askdirectory()
            if folder != "":
                response = []
                nameList = []
                result = QuetTrenFolderML(folder, nameList, response)
                listFile = Listbox(app, height=19, width = 90)
                
                i = 1
                raw = ""
                for res in range(len(response)):
                    if response[res] == 1:
                        listFile.insert(i, "File: " + str(nameList[res]) + " an toàn")
                        raw += str(i) + ". File: " + str(nameList[res]) + " an toàn\n"
                    elif response[res] == 2:
                        listFile.insert(i, "File: " + str(nameList[res]) + " không phải định dạng PE")
                        raw += str(i) + ". File: " + str(nameList[res]) + " không phải định dạng PE\n"
                    else:
                        #md5 = detectLabel(str(nameList[res]))
                        listFile.insert(i, "File: " + str(nameList[res]) + " NGUY HIỂM ")
                        raw += str(i) + ". File: " + str(nameList[res]) + " nguy hiểm\n"
                    i += 1

                hahahoho = raw


                numberOfFile = len(nameList)
                SoVirusThucTe = 0
                SoVirusTimDung = 0
                SoVirusTimSai = 0
                num = 0
                for i in nameList:
                    if  "Virus" in i :
                        SoVirusThucTe += 1
                        if(response[num] == 2):
                            SoVirusThucTe -= 1
                    num += 1

                for i in range(len(nameList)):
                    if "Virus" in nameList[i] and response[i] != 1 and response[i] != 2:
                        SoVirusTimDung += 1
                    elif "Virus" not in nameList[i] and response[i] != 1 and response[i] != 2:
                        SoVirusTimSai += 1

            state = "Trạng thái: đã scan xong " + str(folder) + "\nSố file Virus phát hiện: " + str(SoVirusTimDung) + "\nSố file Virus thực tế: " + str(SoVirusThucTe)# + "\nĐộ chính xác: " float(number)
            state += "\nSố file Virus tìm sai: " + str(SoVirusTimSai)
            state += "\nTổng số file: " + str(numberOfFile)
            if SoVirusTimDung == 0 and SoVirusThucTe == 0 and SoVirusTimSai == 0:
                state += "\nĐộ chính xác: 100%"
            elif SoVirusTimDung == 0 and SoVirusThucTe == 0 and SoVirusTimSai != 0:
                state += "\nĐộ chính xác: 0%"
            elif SoVirusTimDung-SoVirusTimSai == SoVirusThucTe:
                state += "\nĐộ chính xác: 100%"
            elif (SoVirusThucTe != 0 and SoVirusTimDung-SoVirusTimSai == 0):
                state += "\nĐộ chính xác: 0%"
            elif (SoVirusTimDung-SoVirusTimSai == 0 and SoVirusThucTe != 0):
                state += "\nĐộ chính xác: 0%"
            else:
                if(SoVirusThucTe > SoVirusTimDung-SoVirusTimSai):
                    msg = ((SoVirusTimDung-SoVirusTimSai)/SoVirusThucTe)*100
                    state = state +  "\nĐộ chính xác: " + str(msg) + "%"
                else:
                    msg = (SoVirusThucTe/(SoVirusTimDung-SoVirusTimSai))*100
                    state = state +  "\nĐộ chính xác: " + str(msg) + "%"

            messagebox.showinfo("Kết quả", state)


class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        self.title("Quét mã độc trên Window")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="KMA ANTIVIRUS",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Trích xuất ra JSON",text_font=('bold',11),
                                                command=self.trichXuatClass)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Trích chọn đặc trưng",text_font=('bold',11),
                                                command=openTrichWindow)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        self.button_3 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Mô hình học máy học sâu",text_font=('bold',11),
                                                command=TaoClassifier)
        self.button_3.grid(row=4, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Chế độ màu:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=3, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="" ,
                                                   height=300,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

        # ============ frame_right ============

        self.label_2 = customtkinter.CTkLabel(master=self.frame_right,
                                              text="Trạng thái:",
                                              text_font=('bold',12))  # font name and size in px
        self.label_2.grid(row=4, column=0)

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Quét Folder",text_font=('bold',11),
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.QuetFolderClass)
        self.button_5.grid(row=5, column=0, columnspan=1, pady=20, padx=20, sticky="we")

        self.button_6 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Quét File",text_font=('bold',11),
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=QuetVirus)
        self.button_6.grid(row=5, column=1, columnspan=1, pady=20, padx=20, sticky="we")

        self.button_7 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Credit",text_font=('bold',11),
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.cre)
        self.button_7.grid(row=5, column=2, columnspan=1, pady=20, padx=20, sticky="we")
        # set default values
        self.optionmenu_1.set("Dark")

    def button_event(self):
        print("Button pressed")


    def trichXuatClass(self):
        trichXuat()
        self.label_info_1.configure(text = hahahoho)


    def QuetFolderClass(self):
        QuetVirusFolder()
        self.label_info_1.configure(text = hahahoho)
        #print("\n\n============================\n" + hahahoho + "\n\n============================\n")

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.destroy()

    def cre(self):
        tkinter.messagebox.showinfo("CREDIT", "Antimalware owned by the faculty of IT - ACT. Representative: Thuanld, Hungnd")




if __name__ == "__main__":
    app = App()
    app.mainloop()
