from ultralytics import YOLO
import cv2

#importação da biblioteca YOLO do ultralytcs para detecção de objetos e OpenCV para manipulação de imagens


#carregar modelos 
model = YOLO('yolov8s.pt')
#yolov8n.pt  -menor 
#yolov8s.pt
#yolov8m.pt
#yolov8l.pt
#yolov8x.pt  -maior

#inicialização da instancia do modelo YOLO com arquivo de pesos com 'yolov8x.pt'


#capturar video da webcan
cap = cv2.VideoCapture(0) # 0 para webcan padrao, altere se tiver varias webcans
cap.set(cv2.CAP_PROP_FPS, 60) # Tenta definir o FPS

frame_nmr = -1
ret = True
while ret:
    frame_nmr +=1
    ret, frame = cap.read()
    if not ret: 
        print("falha ao capturar webcan")
        break
    
    #detecção de veiculos
    deteccoes = model(frame, verbose=False)[0]
    
    #exibir quadro 
    cv2.imshow('Detecções', deteccoes.plot())
    
    #pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#liberar webcan e fechar todas as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()
