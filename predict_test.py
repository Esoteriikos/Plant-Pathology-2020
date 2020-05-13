import csv
import models
from load_process_data import data_from_id

images_path = r'plant-pathology-2020-fgvc7\images\\'
test_path = r"plant-pathology-2020-fgvc7\test.csv"

sub_path = r"output2.csv"

model = models.model_efn()
model.load_weights(r'trained_model\modelv2-1-27.h5')

print("Loading Test data")

test = data_from_id(images_path=images_path, csv_path=test_path, img_shape=(150, 150))
x_test = test.image_target_array(image_label='image_id', random_state=-1)

x_test = x_test/255

y_pred = model.predict(x_test)
#y_pred = pd.get_dummies(y_pred)

with open(sub_path, 'w', newline='') as f:
    write = csv.writer(f)
    write.writerow(['image_id','healthy','multiple_diseases','rust','scab'])
    
    for i in range(len(x_test)):
        x='Test_'+str(i)
        a = str(y_pred[i][0])
        b = str(y_pred[i][1])
        c = str(y_pred[i][2])
        d = str(y_pred[i][3])
        write.writerow([x, a, b, c, d])
