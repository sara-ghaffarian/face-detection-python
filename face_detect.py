import cv2

#آماده سازی فایل تشخیص چهره
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#خواندن تصویر
img=cv2.imread('test.jpg')

#تبدیل تصویر رنگی به تصویر خاکستری برای تشخیص چهره
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# مرحله ی تشخیص چهره های موجود در عکس
# scaleFactor = 1.1 → هر بار تصویر 10٪ کوچک می‌شود تا چهره‌ها در اندازه‌های مختلف شناسایی شوند
# minNeighbors = 4 → حداقل ۴ مستطیل باید همپوشانی داشته باشند تا یک چهره واقعی تشخیص داده شود
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# کشیدن مستطیل آبی دور چهره‌ها با ضخامت دو
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ذخیره تصویر خروجی
cv2.imwrite('result.jpg', img)
print("صورت های موجود در عکس تشضخیص داده شد و در فایل result.jpg ذخیره شد")


#  برای تشخیص چهره در ویدیو فریم به فریم پیش میریم

cap = cv2.VideoCapture(0) 
print("برای خروج کلید Q را بزنید...")

while True:
    # خواندن فریم
    ret, frame = cap.read()
    if not ret:
        break 
    
    # تبدیل فریم به خاکستری
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تشخیص چهره‌ها در فریم
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    # کشیدن مستطیل سبز دور چهره‌ها با ضخامت 2
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    
    #برای خروج از حلقه Q را بزنید
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

# آزادسازی ویدیو و بستن همه پنجره‌ها
cap.release()
cv2.destroyAllWindows()