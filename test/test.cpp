#include <array>
#include <conio.h>
#include <myo/myo.hpp>
#include <GRT/GRT.h>
#include <sapi.h>
#include <string>

using namespace std;
using namespace GRT;

std::wstring s2ws(const std::string& s)
{
	int len;
	int slength = (int)s.length() + 1;
	len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
	std::wstring r(buf);
	delete[] buf;
	return r;
}

//数据采集
class DataCollector : public myo::DeviceListener {
public:
	DataCollector()
		:onArm(false), isUnlocked(false), currentPose(), threshold(0), accuracy(0)
		, fft(2048, 1, 8, true, true), svm(SVM::LINEAR_KERNEL)
		, maf(15, 8), lpf(0.1, 1, 8, 200, 1.0 / 500.0)
	{
		//存储向量初始化
		emgData = vector<vector<int>>(8, vector<int>(0, 0));
		accelData = vector<vector<double>>(3, vector<double>(0, 0));
		orientData = vector<vector<int>>(3, vector<int>(0, 0));
		filteredData = vector<vector<double>>(8, vector<double>(0, 0));
		featureData = vector<double>(0, 0);

		//采样数据初始化
		emgSamples.fill(0);
		accelSamples.fill(0);
		orientSamples.fill(0);

		//训练数据初始化
		accuracy = 0;
		trainingData.setNumDimensions(32);
		trainingData.setDatasetName("DummyData");
		trainingData.setInfoText("This data contains some dummy timeseries data");

		//其他
		svm.getScalingEnabled();
		//svm.enableNullRejection(true);
	}

	//当Myo没有配对时，被调用
	void onUnpair(myo::Myo* myo, uint64_t timestamp) {
		emgSamples.fill(0);
		accelSamples.fill(0);
		orientSamples.fill(0);
		onArm = false;
		isUnlocked = false;
	}

	//当Myo提供新的EMG数据时，被调用
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg) {
		for (int i = 0; i < 8; i++)
			emgSamples[i] = static_cast<int>(emg[i]);

	}

	//当Myo提供新的加速度计数据时，单位G，被调用
	void onAccelerometerData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float>& accel) {
		for (size_t i = 0; i < 3; i++)
			accelSamples[i] = accel[i];

	}

	//当Myo提供新的定位数据时，以四元数表示，被调用
	void onOrientationData(myo::Myo* myo, uint64_t timestamp,
		const myo::Quaternion<float>& rotation) {
		double roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
			1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
		double pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));
		double yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
			1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));
		orientSamples[0] = static_cast<int>((roll + (float)M_PI) / (M_PI * 2.0f) * 18);
		orientSamples[1] = static_cast<int>((pitch + (float)M_PI / 2.0f) / M_PI * 18);
		orientSamples[2] = static_cast<int>((yaw + (float)M_PI) / (M_PI * 2.0f) * 18);
	}

	//当Myo检测到用户动作改变时，被调用
	void onPose(myo::Myo* myo, uint64_t timestamp, myo::Pose pose) {
		currentPose = pose;
		if (pose != myo::Pose::unknown && pose != myo::Pose::rest) {
			myo->unlock(myo::Myo::unlockHold);
			myo->notifyUserAction();
		}
		else {
			myo->unlock(myo::Myo::unlockTimed);
		}
	}

	//当Myo检测到在手臂上时，被调用
	void onArmSync(myo::Myo* myo, uint64_t timestamp, myo::Arm arm,
		myo::XDirection xDirection, float rotation,
		myo::WarmupState warmupState) {
		onArm = true;
		whichArm = arm;
	}

	//当Myo从手臂上移开或则移动时，被调用
	void onArmUnsync(myo::Myo* myo, uint64_t timestamp) {
		onArm = false;
	}

	//当Myo解锁时，被调用
	void onUnlock(myo::Myo* myo, uint64_t timestamp) {
		isUnlocked = true;
	}

	//当Myo锁定时，被调用
	void onLock(myo::Myo* myo, uint64_t timestamp) {
		isUnlocked = false;
	}

	//接收数据存储到向量中
	bool recData(size_t windowSize) {
		for (size_t i = 0; i < 8; i++)
			emgData[i].push_back(emgSamples[i]);
		for (size_t i = 0; i < 3; i++) {
			accelData[i].push_back(accelSamples[i]);
			orientData[i].push_back(orientSamples[i]);
		}
		if (emgData[0].size() == windowSize) {
			return true;
		}
		return false;
	}

	//设置起点手势阈值
	bool setThreshold(size_t windowSize) {
		for (size_t i = 0; i < 8; i++)
			emgData[i].push_back(emgSamples[i]);
		for (size_t i = 0; i < 3; i++) {
			accelData[i].push_back(accelSamples[i]);
			orientData[i].push_back(orientSamples[i]);
		}
		if (emgData[0].size() == windowSize) {
			for (size_t i = 0; i < emgData[0].size(); i++) {
				VectorDouble inputVector;
				for (size_t j = 0; j < emgSamples.size(); j++)
					inputVector.push_back(emgData[j][i]);
				VectorDouble filteredValue = lpf.filter(maf.filter(inputVector));
				for (size_t m = 0; m < filteredValue.size(); m++)
					filteredData[m].push_back(filteredValue[m]);
			}
			for (size_t j = 0; j < 8; j++) {
				double temp = 0;
				for (size_t i = 0; i < filteredData[0].size(); i++) 
					temp += fabs(filteredData[j][i]);
				temp = temp / filteredData[0].size();
				featureData.push_back(temp);
			}
			for (size_t i = 0; i < 8; i++)
				threshold += featureData[i];
			cout << "Set threshold ok!" << endl;
			return true;
		}
		return false;
	}

	//数据预处理
	void preProcessingData() {
		for (size_t i = 0; i < emgData[0].size(); i++) {
			VectorDouble inputVector;
			for (size_t j = 0; j < emgSamples.size(); j++)
				inputVector.push_back(emgData[j][i]);
			VectorDouble filteredValue = lpf.filter(maf.filter(inputVector));
			for (size_t m = 0; m < filteredValue.size(); m++)
				filteredData[m].push_back(filteredValue[m]);
		}
	}

	//特征提取数据(维数变化）
	void featureExtractionData() {
		//计算MAV
		for (size_t j = 0; j < 8; j++) {
			array<double, 4> temp = { 0,0,0,0 };
			for (size_t i = 0; i < filteredData[0].size(); i++) {
				temp[0] += fabs(filteredData[j][i]);
				if (i < filteredData[0].size() - 1) {
					double m = -(filteredData[j][i] * filteredData[j][i + 1]);
					double n = fabs(filteredData[j][i] - filteredData[j][i + 1]);
					if (m > 0 && n >= 0.02)
						temp[1] += 1;
				}

				if (i < filteredData[0].size() - 2) {
					double m = (filteredData[j][i + 1] - filteredData[j][i])*(filteredData[j][i + 1] - filteredData[j][i + 2]);
					if (m >= 0.02)
						temp[2] += 1;
				}

				if (i < filteredData[0].size() - 1) {
					temp[3] += fabs(filteredData[j][i + 1] - filteredData[j][i]);
				}

			}
			temp[0] = temp[0] / filteredData[0].size();
			for(size_t i = 0; i < temp.size(); i++)
				featureData.push_back(temp[i]);
		}
	}

	//将各动作的各个通道的MAV之和发送到文件
	void sumMAVToFile(string mavDataFile,size_t num) {
		ofstream outfile(mavDataFile, ios::app);
		cout << "Opened file : " << mavDataFile << endl;
		double temp = 0;
		for (size_t i = 0; i < 8; i++) 
			temp += featureData[i * 4];
		if (num == 0)
			outfile << temp << endl;
		else
			outfile << temp << ",";
	}

	//特征数据发送到文件（CSV文件，该文件可以直接被分类器使用）
	void featureDataToFile(string trainingDataFile, UINT gestureLabel) {
		ofstream outfile(trainingDataFile, ios::app);
		cout << "Opened file : " << trainingDataFile << endl;
		outfile << gestureLabel << ",";
		for (size_t j = 0; j < featureData.size() - 1; j++)
			outfile << featureData[j] << ",";
		outfile << featureData[featureData.size() - 1] << endl;
		cout << "Done Writing " << trainingDataFile << endl;
		outfile.close();
	}

	//将所有原始数据发送到文件中（CSV文件，该文件可以直接被分类器使用）
	void allDataToFile(string alltrainingDataFile, UINT gestureLabel) {
		ofstream outfile(alltrainingDataFile, ios::app);
		cout << "Opened file" << endl;
		for (size_t i = 0; i < accelData[0].size(); i++) {
			outfile << gestureLabel << ",";
			for (size_t j = 0; j < 8; j++) {
				outfile << emgData[j][i] << ",";
			}
			for (int j = 0; j < 3; j++) {
				outfile << accelData[j][i] << ",";
			}
			for (int j = 0; j < 2; j++) {
				outfile << orientData[j][i] << ",";
			}
			outfile << orientData[2][i] << endl;
		}
		cout << "Done writing " << alltrainingDataFile << endl;
		cout << "Total size : " << accelData[0].size() << endl;
		outfile.close();
		cout << "Closing file" << endl;
	}

	//将EMG数据发送到文件中（CSV文件，该文件可以直接被分类器使用）
	void emgDataToFile(string emgTrainingDataFile, UINT gestureLabel) {
		ofstream outfile(emgTrainingDataFile, ios::app);
		ofstream file("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\fftData.csv", ios::app);
		cout << "Opened file" << endl;

		//将滤波后的EMG数据发送到文件
		for (size_t i = 0; i < emgData[0].size(); i++) {
			outfile << gestureLabel << ",";
			VectorDouble inputVector;
			for (size_t j = 0; j < 8; j++) {
				inputVector.push_back(emgData[j][i]);
			}
			VectorDouble filteredValue1 = maf.filter(inputVector);
			VectorDouble filteredValue2 = lpf.filter(filteredValue1);
			if (!fft.update(filteredValue2)) {
				cout << "Fast Fourier Transform failed!" << endl;
				cin.ignore();
				exit(EXIT_FAILURE);
			}
			for (size_t m = 0; m < filteredValue2.size() - 1; m++)
				outfile << filteredValue2[m] << ",";
			outfile << filteredValue2[filteredValue2.size() - 1] << endl;
		}

		//将滤波后的EMG数据快速傅里叶变换后发送到文件
		vector<FastFourierTransform> fftResults = fft.getFFTResults();
		vector<vector<double>> temp = vector<vector<double>>(fftResults.size(), vector<double>(0, 0));
		for (size_t m = 0; m < fftResults.size(); m++) {
			VectorDouble magnitudeData = fftResults[m].getMagnitudeData();
			for (size_t n = 0; n < magnitudeData.size(); n++)
				temp[m].push_back(magnitudeData[n]);
		}
		for (size_t j = 0; j < temp[0].size(); j++) {
			file << gestureLabel << ",";
			for (size_t i = 0; i < fftResults.size() - 1; i++) {
				file << temp[i][j] << ",";
			}
			file << temp[fftResults.size() - 1][j] << endl;
		}


		cout << "Done writing " << emgTrainingDataFile << endl;
		cout << "Total size : " << emgData[0].size() << endl;
		file.close();
		outfile.close();
		cout << "Closing file" << endl;
	}

	//将加速度数据发送到文件中（CSV文件，该文件可以直接被分类器使用）
	void accelDataToFile(string accelTrainingDataFile, UINT gestureLabel) {
		ofstream outfile(accelTrainingDataFile, ios::app);
		cout << "Opened file" << endl;
		for (size_t i = 0; i < accelData[0].size(); i++) {
			outfile << gestureLabel << ",";
			for (size_t j = 0; j < 2; j++) {
				outfile << accelData[j][i] << ",";
			}
			outfile << accelData[2][i] << endl;
		}
		cout << "Done writing " << accelTrainingDataFile << endl;
		cout << "Total size : " << accelData[0].size() << endl;
		outfile.close();
		cout << "Closing file" << endl;
	}

	//清空当前数据存储向量
	void clearData() {
		featureData.clear();
		for (size_t i = 0; i < 8; i++) {
			emgData[i].clear();
			filteredData[i].clear();
		}
		for (size_t i = 0; i < 3; i++) {
			accelData[i].clear();
			orientData[i].clear();
		}
		predictClassLabel = 0;
	}

	//类标签设置
	UINT classLabelFile(string gestureName) {
		UINT gestureLabel = 1;
		UINT num;
		string tmp;
		map<string, UINT> gesture;

		//查找ClassIDFile.txt下，是否含有gesturename
		fstream f("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::in);
		while (f.peek() != EOF) {
			f >> tmp;
			f >> num;
			gesture[tmp] = num;
		}
		f.close();

		//给每个手势编号（从1开始）
		if (gesture.size() == 0) {
			gestureLabel = 1;
			gesture[gestureName] = gestureLabel;
			f.open("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::app);
			f << gestureName << " ";
			f << gesture[gestureName] << endl;
			f.close();
			return gestureLabel;
		}
		if (gesture.count(gestureName)) {
			gestureLabel = gesture[gestureName];
			return gestureLabel;
		}
		else {
			gestureLabel = gesture.size() + 1;
			gesture[gestureName] = gestureLabel;
			f.open("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::app);
			f << gestureName << " ";
			f << gesture[gestureName] << endl;
			f.close();
			return gestureLabel;
		}
	}

	//开始手形手势训练
	void startTraining(string trainingDataFile, string trainingModelFile) {
		if (!trainingData.loadDatasetFromCSVFile(trainingDataFile)) {
			cout << "ERROR: Failed to load file to dataset!" << endl;
			exit(EXIT_FAILURE);
		}
		cout << "getNumDimensions : " << trainingData.getNumDimensions() << endl;
		testData = trainingData.split(90);


		cout << "Ready to train data!" << endl;
		pipeline.setClassifier(svm);
		if (!pipeline.train(trainingData)) {
			cout << "Filed to train classfier!" << endl;
			cin.ignore();
			exit(EXIT_FAILURE);
		}
		if (!pipeline.save(trainingModelFile)) {
			cout << "Failed to save the classifier model!" << endl;
			exit(EXIT_FAILURE);
		}
		cout << "Ready to test data!" << endl;

		if (pipeline.test(testData)) {
			accuracy = pipeline.getTestAccuracy();
			UINT predictedClassLabel = pipeline.getPredictedClassLabel();
			cout << "Accuracy : " << accuracy << endl;
		}
		else
			exit(EXIT_FAILURE);
	}

	//获取手势类标签
	void getClassLabel(string classLabelFile) {
		ifstream fin(classLabelFile);
		UINT i;
		string tmp;
		while (fin >> tmp) {
			fin >> i;
			gestureNames[i] = tmp; 
		}
	}

	//手势识别
	void gestureRecognition(string trainingModelFile) {
		if (!pipeline.load(trainingModelFile)) {
			cout << "Filed to load the classfier model!" << endl;
			exit(EXIT_FAILURE);
		}
		double temp = 0;
		for (size_t i = 0; i < 8; i++)
			temp += featureData[i * 4];
		if (temp >= threshold) {
			if (!pipeline.predict(featureData)) {
				cout << "Filed to perform prediction!" << endl;
				cin.ignore();
				exit(EXIT_FAILURE);
			}
			//if (GetAsyncKeyState(VK_SPACE)) {
				predictClassLabel = pipeline.getPredictedClassLabel();
				double maximumLikelihood = pipeline.getMaximumLikelihood();
				if (predictClassLabel)
					cout << maximumLikelihood << " " << gestureNames[predictClassLabel] << endl;
			//}
		}
	}

	//文本转语音
	void textSpeak() {
		ISpVoice * pVoice = NULL;
		if (FAILED(::CoInitialize(NULL)))
			exit(EXIT_FAILURE);
		HRESULT hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&pVoice);
		if (SUCCEEDED(hr))
		{
			wstring temp = s2ws(gestureNames[predictClassLabel]);
			LPCWSTR result = temp.c_str();
			hr = pVoice->Speak(result, 0, NULL);
			pVoice->Release();
			pVoice = NULL;
		}
		::CoUninitialize();
	}

	//打印EMG数据和pitch,roll,yaw
	void print() {
		cout << '\r';
		//EMG数据
		for (size_t i = 0; i < emgSamples.size(); i++) {
			ostringstream oss;
			oss << static_cast<int>(emgSamples[i]);
			string emgString = oss.str();
			cout << '[' << emgString << string(4 - emgString.size(), ' ') << ']';
		}
		//pitch,roll,yaw
		cout << '[' << string(orientSamples[0], '*') << string(18 - orientSamples[0], ' ') << ']'
			<< '[' << string(orientSamples[1], '*') << string(18 - orientSamples[1], ' ') << ']'
			<< '[' << string(orientSamples[2], '*') << string(18 - orientSamples[2], ' ') << ']';
		if (onArm) {
			string poseString = currentPose.toString();

			cout << '[' << (isUnlocked ? "unlocked" : "locked  ") << ']'
				<< '[' << (whichArm == myo::armLeft ? "L" : "R") << ']'
				<< '[' << poseString << string(14 - poseString.size(), ' ') << ']';
		}
		else {
			cout << '[' << std::string(8, ' ') << ']' << "[?]" << '[' << string(14, ' ') << ']';
		}
		cout << flush;
	}


	//存储数据
	vector<vector<int>> emgData;
	vector<vector<double>> accelData;
	vector<vector<int>> orientData;

	//MYO状态
	bool onArm;
	bool isUnlocked;
	myo::Arm whichArm;
	myo::Pose currentPose;

	//采样数据
	array<int, 8> emgSamples;
	array<double, 3> accelSamples;
	array<int, 3> orientSamples;

	//用于训练和手势识别
	VectorDouble featureData;
	double threshold;
	double accuracy;
	UINT predictClassLabel;
	SVM svm;
	GestureRecognitionPipeline pipeline;
	vector<vector<double>> filteredData;
	ClassificationData trainingData;
	ClassificationData testData;
	map<UINT, string> gestureNames;

	//工具
	FFT fft;
	LowPassFilter lpf;
	MovingAverageFilter maf;
};

int main(int argc, const char* argv[]) {
	while (1) {
		cout << "Function Menu" << endl;
		cout << "1.CollectData" << endl << "2.TrainModel" << endl << "3.GestureRecognition" << endl;
		char ans;
		string gesturename;
		size_t n;
		UINT gestureLabel;
		DataCollector collector;

		while ((ans = _getch()) != '1' && ans != '2' && ans != '3');

		switch (ans) {
		case '1':
			try {
				myo::Hub hub("com.example.gesturemyo");
				cout << "Attempting to find a Myo ..." << endl;
				myo::Myo* myo = hub.waitForMyo(10000);
				if (!myo) {
					throw runtime_error("Unable to find a Myo!");
				}
				cout << "Connected to a Myo armband!" << endl << endl;
				myo->setStreamEmg(myo::Myo::streamEmgEnabled);
				hub.addListener(&collector);
				system("md C:\\Users\\LiuYu\\Desktop\\Data\\RawData");
				cout << "Enter the name of gesture : ";
				cin >> gesturename;
				if (gesturename.size() == 0) return 0;
				cout << "Enter the number of recording : ";
				cin >> n;
				cout << "Press h when ready to start recording." << endl;
				while ('h' != _getch()) {
				}
				cout << "Recording gesture. Press Esc to end recording." << endl;
				while (n--) {
					hub.run(2);
					if (!collector.onArm) 
						throw runtime_error("!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!");
					while (!collector.recData(200)) {
						hub.run(2);
						collector.print();
					}
					if (GetAsyncKeyState(VK_ESCAPE))
						break;
					cout << "Use recording? (y/n)" << endl;
					char ch;
					while ((ch = _getch()) != 'n' && ch != 'y');
					if (ch == 'y') {
						collector.preProcessingData();
						collector.featureExtractionData();
						gestureLabel = collector.classLabelFile(gesturename);
						collector.featureDataToFile("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\trainingData.csv", gestureLabel);
						collector.sumMAVToFile("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\sumMAVData.csv",n);
					}
					collector.clearData();
				}
				collector.clearData();
			}
			catch (const std::exception& e) {
				cerr << "Error: " << e.what() << endl;
				break;
			}
			break;
		case '2':
			system("md C:\\Users\\LiuYu\\Desktop\\Data\\Model");
			collector.startTraining("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\trainingData.csv", "C:\\Users\\LiuYu\\Desktop\\Data\\Model\\SVMModel.txt");
			break;
		case '3':
			try {
				myo::Hub hub("com.example.gesturemyo");
				cout << "Attempting to find a Myo ..." << endl;
				myo::Myo* myo = hub.waitForMyo(10000);
				if (!myo) {
					throw runtime_error("Unable to find a Myo!");
				}
				cout << "Connected to a Myo armband!" << endl << endl;
				myo->setStreamEmg(myo::Myo::streamEmgEnabled);
				hub.addListener(&collector);
				cout << "Please relax your hand to set the threshold!" << endl;
				while (!collector.setThreshold(2000)) {
					hub.run(2);
					if (!collector.onArm)
						throw runtime_error("!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!");
				}
				collector.clearData();
				collector.getClassLabel("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt");
				while (1) {
					hub.run(2);
					if (!collector.onArm)
						throw runtime_error("!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!");
					while (!collector.recData(100)) 
						hub.run(2);
					collector.preProcessingData();
					collector.featureExtractionData();
					collector.gestureRecognition("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\SVMModel.txt");
					collector.textSpeak();
					collector.clearData();
					if (GetAsyncKeyState(VK_ESCAPE))
						break;
				}
				collector.clearData();
			}
			catch (const std::exception& e) {
				cerr << "Error: " << e.what() << endl;
				break;
			}
			break;
		default:
			break;
		}
	}
	return 0;
}