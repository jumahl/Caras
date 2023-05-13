import cv2

def detectar_cara():
    cascada_cara = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camara = cv2.VideoCapture(0)

    while True:
        ret, frame = camara.read()
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = cascada_cara.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in caras:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Deteccion de Caras', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detectar_cara()
    