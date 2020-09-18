# from sklearn.preprocessing import MinMaxScaler
# 在keras1.0到2.0有些许变化，主要是output_dim -> units,dropout_U -> recurrent_dropout 
# from trainingset_selection import TrainingSetSelection
from keras.models import Sequential, load_model, save_model
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from preprocessing import get_ids_and_files_in_dir, MinMaxScaler, NormalDistributionScaler, create_interval_month_dataset, invert_MinMaxScaler, invert_MinMaxScaler1, minMaxscaler_list, get_NSE
# from preprocessing import percentile_remove_outlier, binning_date_y
import os, shutil
import keras
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import time

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        book = openpyxl.load_workbook('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\loss.xlsx')
        writer = pd.ExcelWriter('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\loss.xlsx', engine='openpyxl') 
        writer.book = book
        pre_datap = pd.DataFrame(self.losses[loss_type], columns=['train_loss'])
        true_datap = pd.DataFrame(self.val_loss[loss_type], columns=['valify_loss'])
        df = pd.concat([true_datap,pre_datap], axis=1)
        df.to_excel(writer, 'loss')
        writer.close()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        if not os.path.exists('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\fig'):
            os.makedirs('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\fig')
        plt.savefig('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\fig\\1.png')
        plt.close()
        # plt.show()


class NeuralNetwork():
    def __init__(self,
                 training_set_dir,
                 model_save_dir,
                 output_dir,
                 training_set_id_range,
                 training_set_length,
                 epochs,
                 batch_size,
                 history_loss,
                 model_file_prefix='model',
                 scaler = 'mm',
                 **kwargs):
        """
        :param training_set_dir: directory contains the training set files. File format: 76.csv
        :param model_save_dir: directory to receive trained model and model weights. File format: model-76.json/model-weight-76.h5
        :param model_file_prefix='model': file prefix for model file
        :param training_set_range=(0, np.Inf): enterprise ids in this range (a, b) would be analyzed. PS: a must be less than b
        :param training_set_length=3: first kth columns in training set file will be used as training set and the following one is expected value
        :param train_test_ratio=3: the ratio of training set size to test set size when splitting input data
        :param output_dir=".": output directory for prediction files
        :param scaler: scale data set using - mm: MinMaxScaler, norm: NormalDistributionScaler  数据归一化手段
        :param **kwargs: lstm_output_units=4: output dimension of LSTM layer;
                        activation_lstm='relu': activation function for LSTM layers;
                        activation_dense='relu': activation function for Dense layer;
                        activation_last='softmax': activation function for last layer;
                        drop_out=0.2: fraction of input units to drop;
                        np_epoch=25, the number of epoches to train the model. epoch is one forward pass and one backward pass of all the training examples;
                        batch_size=100: number of samples per gradient update. The higher the batch size, the more memory space you'll need;
                        loss='categorical_crossentropy': loss function;
                        optimizer='rmsprop'
        """
        self.training_set_dir = training_set_dir
        self.model_save_dir = model_save_dir
        self.model_file_prefix = model_file_prefix
        self.training_set_id_range = training_set_id_range
        self.training_set_length = training_set_length
        self.output_dir = output_dir
        self.history_loss = history_loss
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = kwargs.get('test_size', 0.2)
        self.lstm_output_units = kwargs.get('lstm_output_units', 50)
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')       #sigmoid  tanh
        self.activation_dense = kwargs.get('activation_dense', 'relu')
        self.activation_last = kwargs.get('activation_last', 'relu')    # softmax for multiple output （linear）
        self.dense_layer = kwargs.get('dense_layer', 2)  # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 2) # at least 2 layers
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.optimizer = kwargs.get('optimizer', 'adam') # 原为rmsprop


    def NN_model_train(self, trainX, trainY, testX, testY, model_save_path, history_loss):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: expect value of test data
        :param model_save_path: h5 file to store the trained model
        :param override: override existing models
        :return: model after training
        """
        input_dim = trainX.shape[1]
        # 直接定为1了，比较简单
        units = 1
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfit
        # 对input_shape，3个参数，第一个batch_size可以省略，第二个表示输入集的维度，即由几个数预测，如这里是3-1，第3个表训练次数
        # model.add(LSTM(units=self.lstm_output_units,
        #                input_shape=(trainX.shape[1], trainX.shape[2]),
        #                activation=self.activation_lstm,
        #                recurrent_dropout=self.drop_out,
        #                return_sequences=True))
        # for i in range(self.lstm_layer-2):
        #     model.add(LSTM(units=self.lstm_output_units,
        #                activation=self.activation_lstm,
        #                recurrent_dropout=self.drop_out,
        #                retc = np.column_stack((a, b))urn_sequences=True ))
        # return sequences should be False to avoid dim error when concatenating with dense layer
        # model.add(LSTM(units=self.lstm_output_units, activation=self.activation_lstm, recurrent_dropout=self.drop_out))
        # applying a full connected NN to accept output from LSTM layer
        # for i in range(self.dense_layer-1):
        #     model.add(Dense(units=self.lstm_output_units, activation=self.activation_dense))
        #     model.add(Dropout(self.drop_out))
        # model.add(Dense(units=units, activation=self.activation_last))
        # configure the learning process
        model = Sequential()
        model.add(LSTM(units=self.lstm_output_units, activation=self.activation_last,input_dim = trainX.shape[2], recurrent_dropout=self.drop_out, return_sequences=True))
        model.add(LSTM(units=self.lstm_output_units, activation=self.activation_last, recurrent_dropout=self.drop_out))
        model.add(Dense(units))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        # verbose=0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
        model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, validation_data=(testX, testY), verbose=0, shuffle=False, callbacks=[history_loss])
        # plt.plot(LSTM.LSTM['loss'], label='train')
        # plt.plot(LSTM.LSTM['val_loss'], label='valid')
        # model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        # model.fit(x=trainX, y=trainY, epochs=self.epochs, batch_size=self.batch_size, validation_data=(testX, testY))
        # score = model.evaluate(trainX, trainY, self.batch_size)
        # print("Model evaluation: {}".format(score))
        # store model to json file
        save_model(model, model_save_path)


    @staticmethod
    def NN_prediction(dataset, model_save_path):
        dataset = np.asarray(dataset)
        if not os.path.exists(model_save_path):
            raise ValueError("Lstm model not found! Train one first or check your input path: {}".format(model_save_path))
        model = load_model(model_save_path)
        # predict_class = model.predict_classes(dataset)
        # class_prob = model.predict_proba(dataset)
        predict_data = model.predict(dataset)
        # return predict_class, class_prob
        return predict_data

    # 输入数据，得到需要预测的月份数据集和需要使用的月份模型
    def create_raw_set(self, dataset, look_back, model_save_path_list):
        a = len(dataset)%12
        model_save_path = model_save_path_list[a]
        raw_input_set = dataset[-look_back:]
        return raw_input_set, model_save_path



    def get_data(self, input_file_regx):
        """
        :param override=Fasle: rerun the model prediction no matter if the expected output file exists
        :return: model file, model weights files, prediction file, discrepancy statistic bar plot file
        """
        # get training sets for lstm training
        # print("Scanning files within select id range ...")
        ids, files = get_ids_and_files_in_dir(inputdir=self.training_set_dir,
                                                          range=self.training_set_id_range,
                                                          input_file_regx=input_file_regx)
        print("Scanning done! Selected enterprise ids are {}".format(ids))
        if not files:
            raise ValueError("No files selected in current id range. Please check the input training set directory, "
                            "input enterprise id range or file format which should be '[0-9]+.csv'")

        # get train, test, validation data
        for id_index, id_file in enumerate(files):
            file_id = ids[id_index]
            # store prediction result to prediction directory
            enter_file = self.training_set_dir + "\\" + id_file
            print("Processing dataset - enterprise_id is: {}".format(file_id))
            # print("Reading from file {}".format(enter_file))
            df = pd.read_csv(enter_file, header=0, names=['PE'])
            # df.index = range(len(df.index))
            # retrieve training X and Y columns. First column is customer_id
            # 这是数据的输入，可以自己改数据
            # select_col = ['customer_id']
            # select_col = np.append(select_col, ['X' + str(i) for i in range(1, 1+self.training_set_length)])
            # select_col = np.append(select_col, ['Y', 'enterprise_id'])
            # df_selected = df.loc[:, select_col]#iloc需要数字索引，ix已被删除
            dfList = []
            for Pe in df.values.tolist():
                dfList += Pe
        return dfList, file_id  #只打开一个文件
            # remove outlier records对于蒸发数据，选取合格的行会舍弃大量数据，降低数据的量故舍去四分位检测
            # df_selected = percentile_remove_outlier(df_selected, filter_start=1, filter_end=2+self.training_set_length)
            # scale the train columns归一化数据mm表线性函数归一化，norm表正太分布（注意改正）
            #total_sample_count = len(val_predict_class)
            #val_test_label = np.asarray([list(x).index(1) for x in val_test])
            #match_count = (np.asarray(val_predict_class) == np.asarray(val_test_label.ravel())).sum()
            #print("Precision using validation dataset is {}".format(float(match_count) / total_sample_count))

    def get_model(self, dfList, file_id, num_train, override=False):
        dfList = dfList
        file_id = file_id
        all_df_selected = create_interval_month_dataset(dfList, look_back=self.training_set_length)
        loss_short = 0
        list_raw = []
        list_pre = []
        model_save_path_list = []
        for month in range(1, 13):
            # print("Scaling data of month_" + str(month) + "...")
            df_selected = all_df_selected.get('monthSet_'+str(month))
            if self.scaler == 'mm':
                df_scale, minVal, maxVal = MinMaxScaler(df_selected, start_col_index=0, end_col_index=self.training_set_length+1)
            elif self.scaler == 'norm':
                df_scale, meanVal, stdVal = NormalDistributionScaler(df_selected, start_col_index=0, end_col_index=self.training_set_length+1)
            else:
                raise ValueError("Argument scaler must be mm or norm!")
            # bin date y将y值分类，但是对于精确预测而言可能不太合适，故舍去163，164行
            # df_bin, bin_boundary = binning_date_y(df_scale, y_col=1+self.training_set_length, n_group=5)
            # print("Bin boundary is {}".format(bin_boundary))
            df_bin = df_scale
            # get train and test dataset注意asarray和array的区别
            # print("Randomly selecting training set and test set...")
            all_data_x = np.asarray(df_bin.iloc[0:int(len(df_bin.index)), 0:self.training_set_length]).reshape((int(len(df_bin.index)), 1, self.training_set_length))
            # 即转化为3个数组组成的大数组，其中小数组一行，n列
            all_data_y = np.asarray(df_bin.iloc[0:int(len(df_bin.index)), self.training_set_length])
            # test_data_x = np.asarray(df_bin.iloc[int(len(df_bin.index)*0.8):, 0:self.training_set_length]).reshape((len(df_bin.index)-int(len(df_bin.index)*0.8), 1, self.training_set_length))
            # 即转化为3个数组组成的大数组，其中小数组一行，n列
            # test_data_y = np.asarray(df_bin.iloc[int(len(df_bin.index)*0.8):, self.training_set_length])
            # convert y label to one-hot dummy label现在不用one-hot
            # y_dummy_label = np.asarray(pd.get_dummies(all_data_y))
            # format train, test, validation data利用sklearn中的train_test_split函数好用
            # sub_train, val_train, sub_test, val_test = train_test_split(all_data_x, y_dummy_label, test_size=self.test_size)
            # train_x, test_x, train_y, test_y = train_test_split(sub_train, sub_test, test_size=self.test_size)
            train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=self.test_size)
            # create and fit the NN model
            #self.batch_size = len(train_x)
            sub_model_save_path = self.model_save_dir + "\\" + self.model_file_prefix + "-look_back_of-" + str(self.training_set_length) + "-epochs_of-" + str(self.epochs) + "-batch_sizes_of-" + str(self.batch_size) + "-" + str(file_id)
            model_save_path = sub_model_save_path + "of month_" + str(month) + 'num of' + str(num_train) + ".h5"
            # check if model file exists决定是否进行训练
            if month == 1:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-120-02of month_1.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 2:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_2.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 10:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-160-02of month_10.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 12:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-80-02of month_12.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 5:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_5.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 6:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_6.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 7:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_7.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 8:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_8.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 9:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_9.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            elif month == 11:
                model_save_path_be = self.model_save_dir + "\\" + "model-look_back_of-240-epochs_of-1000-batch_sizes_of-180-02of month_11.h5"
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                shutil.copy(model_save_path_be, model_save_path)  #为1月与10月使用同一个精度高模型
            print(os.path.exists(model_save_path))
            if not os.path.exists(model_save_path) or override:
                self.NN_model_train(train_x, train_y, test_x, test_y, model_save_path=model_save_path, history_loss=self.history_loss)
                self.history_loss.loss_plot('epoch')
                png_path1 = 'E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\fig\\1.png'
                png_path2 = 'E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\fig\\' + str(file_id) + '\\L' + str(self.training_set_length) + "-E" + str(self.epochs) + "-B" + str(self.batch_size) + "-" + str(file_id) + "of month_" + str(month) + 'num of' + str(num_train) + '.png'
                if os.path.exists(png_path2):
                    os.remove(png_path2)
                os.rename(png_path1, png_path2)
            model_save_path_list += [model_save_path]
        return model_save_path_list
            # generate prediction for training
            # print("Predicting the output of validation set...")
            # val_predict_class, val_predict_prob = self.NN_prediction(val_train, model_save_path=model_save_path)
            # statistic of discrepancy between expected value and real value
#            val_predict_data = self.NN_prediction(test_data_x, model_save_path=model_save_path)
#            pre_data = invert_MinMaxScaler(val_predict_data, minVal, maxVal)
#            raw_data = invert_MinMaxScaler1(test_data_y, minVal, maxVal)
#            for data_p,data_r in zip(pre_data,raw_data):
#                # print('Predict is ' + str(data_p[0]) + ', raw is ' + str(data_r) + ', loss is ' + str((data_p[0]-data_r)/data_r))
#                # print(str(data_p[0]) + ', ' + str(data_r) + ', ' + str((data_p[0]-data_r)/data_r))
#                loss_short += abs((data_p[0]-data_r)/data_r)
#                list_raw += [data_r]
#                list_pre += [data_p[0]]
#            print("------E" + str(self.epochs) + "------B_" + str(self.batch_size) + "-----------------")
#            print('The loss is ' + str(loss_short) + 'The NSE RMSE MAE is ' + str([get_NSE(list_raw = list_raw, list_pre = list_pre)]))


    def predict_future(self, pre_years, input_file_regx):
        is_get_loss = 1
        pre_years = pre_years
        input_file_regx = input_file_regx
        dfList, file_id = self.get_data(input_file_regx)
        if is_get_loss:
            pre_years = 10
            test_train_data = dfList[:660-12*pre_years]  #去掉减号可以用于预测
        else:
            test_train_data = dfList
        for i in range(pre_years):
            model_save_path_list = self.get_model(test_train_data, file_id, i, override=False)
            for j in range(12):
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ', epoch of ' + str(i))
                raw_input_set, model_save_path_need = self.create_raw_set(test_train_data, look_back=self.training_set_length, model_save_path_list=model_save_path_list)
                print(len(raw_input_set))
                output_scaled_set, minVal, maxVal = minMaxscaler_list(raw_input_set)
                output_scaled_set = np.array(output_scaled_set).reshape(1,1,self.training_set_length)
                need_val_predict_data = self.NN_prediction(output_scaled_set, model_save_path=model_save_path_need)
                need_pre_data = invert_MinMaxScaler1(need_val_predict_data[0], minVal, maxVal)
                test_train_data += need_pre_data.tolist()
        book = openpyxl.load_workbook('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\predict.xlsx')
        writer = pd.ExcelWriter('E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM\\predict.xlsx', engine='openpyxl')
        writer.book = book
        if is_get_loss:
            loss_long = []
            time_loss = pd.Series(pd.period_range('1/1/1960', freq='M', periods=12*55))
            for data_p,data_r in zip(test_train_data,dfList[:660]):
                #for data_p,data_r in zip(test_train_data,dfList[:44*12]):
                loss_long += [(data_p-data_r)/data_r]
            loss_get = pd.DataFrame(loss_long, columns=['loss'])
            nse,rmse,mae = get_NSE(dfList[240:44*12], test_train_data[240:44*12])
            pre_datap = pd.DataFrame(test_train_data, columns=['Pre_data'])
            true_datap = pd.DataFrame(dfList[:660], columns=['True_data'])
            nsep = pd.DataFrame([nse], columns=['NSE'])
            rmsep = pd.DataFrame([rmse], columns=['RMSE'])
            maep = pd.DataFrame([mae], columns=['MAE'])
            df = pd.concat([time_loss,true_datap,pre_datap,loss_get,nsep,rmsep,maep], axis=1)
            df.to_excel(writer, 'EVPval' + '-E' + str(self.epochs) + '-B' + str(self.batch_size))
        else:
            time_loss = pd.Series(pd.period_range('1/1/1960', freq='M', periods=12*(55+pre_years)))
            pre_datap = pd.DataFrame(test_train_data, columns=['Pre_data'])
            true_datap = pd.DataFrame(dfList, columns=['True_data'])
            df = pd.concat([time_loss,true_datap,pre_datap], axis=1)
            df.to_excel(writer, 'EVPpre' + '-E' + str(self.epochs) + '-B' + str(self.batch_size))
        writer.close()


def main():
    training_set_dir = "E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM" #E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM 
    output_dir = "E:\\Learning\\Workspace\\Python\\DeepLearning\\venv\\project\\LSTM"  #D:\\LRuanJian\\Python\\python_work\\DeepLearning\\enev\\Project
    training_set_id_range = (1, 2)
    dense_layer = 2
    model_file_prefix = 'model'
    model_save_dir = output_dir + "\\" + model_file_prefix
    training_set_regx_format = "cluster-(\d+)\.csv"
    training_set_length = 12*20
    pre_years = 40
    stdout_backup = sys.stdout
    log_file_path = output_dir + "\\NN_model_running_log.txt"
    log_file_handler = open(log_file_path, "w")
    print("Log message could be found in file: {}".format(log_file_path))
    #sys.stdout = log_file_handler #因为以二进制打开，故后面print会出错 
    history_loss = LossHistory()
    # 对5 epochs 180左右
    for i in range(26, 27):
        epochs = 1000
        batch_size = 200
        obj_NN = NeuralNetwork(output_dir=output_dir,
                           training_set_dir=training_set_dir,
                           model_save_dir=model_save_dir,
                           model_file_prefix=model_file_prefix,
                           training_set_id_range=training_set_id_range,
                           training_set_length=training_set_length,
                           epochs=epochs,
                           batch_size=batch_size,
                           dense_layer=dense_layer,
                           history_loss=history_loss)
        # record program process printout in log file
        # check if the training set directory is empty. If so, run the training set selection
        # 对不从数据库寻求数据，if语句无意义可以不加
        # if not os.listdir(obj_NN.training_set_dir):
        #     print("Training set files not exist! Run trainingSetSelection.trainingSetGeneration to generate them! Start running generating training set files...")
        #     trainingSetObj = TrainingSetSelection(min_purchase_count=4)
        #     trainingSetObj.trainingset_generation(outdir=obj_NN.training_set_dir)
        #     print("Training set file generation done! They are store at %s directory!".format(obj_NN.training_set_dir))
        # print('Train NN model and test!')
        obj_NN.predict_future(pre_years, input_file_regx=training_set_regx_format)
        # print("Models and their parameters are stored in {}".format(obj_NN.model_save_dir))
        print("------E" + str(epochs) + "------B_" + str(batch_size) + "-----------------")
        # close log file
    print('Finished!')
    log_file_handler.close()
    #sys.stdout = stdout_backup



if __name__ == "__main__":
    main()
