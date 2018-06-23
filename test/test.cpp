#include <array>
#include <conio.h>
#include <myo/myo.hpp>
#include <GRT/GRT.h>

using namespace std;
using namespace GRT;

//���ݲɼ�
class DataCollector : public myo::DeviceListener {
public:
	DataCollector()
		:onArm(false), isUnlocked(false), currentPose(), svm(SVM::LINEAR_KERNEL), fft(1024, 1, 8, true, false), filter(5, 8)
		, maf(5, 8)//, lpf(0.1, 1, 8, 50, 1 / 1000)
	{
		//�洢������ʼ��
		emgData = vector<vector<int>>(8, vector<int>(0, 0));
		accelData = vector<vector<double>>(3, vector<double>(0, 0));
		orientData = vector<vector<int>>(3, vector<int>(0, 0));

		//�������ݳ�ʼ��
		emgSamples.fill(0);
		accelSamples.fill(0);
		orientSamples.fill(0);

		//ѵ�����ݳ�ʼ��
		accuracy = 0;
		trainingData.setNumDimensions(8);
		trainingData.setDatasetName("DummyData");
		trainingData.setInfoText("This data contains some dummy timeseries data");
	}

	//��Myoû�����ʱ��������
	void onUnpair(myo::Myo* myo, uint64_t timestamp) {
		emgSamples.fill(0);
		accelSamples.fill(0);
		orientSamples.fill(0);
		onArm = false;
		isUnlocked = false;
	}

	//��Myo�ṩ�µ�EMG����ʱ��������
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg) {
		for (int i = 0; i < 8; i++)
			emgSamples[i] = static_cast<int>(emg[i]);

	}

	//��Myo�ṩ�µļ��ٶȼ�����ʱ����λG��������
	void onAccelerometerData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float>& accel) {
		for (size_t i = 0; i < 3; i++)
			accelSamples[i] = accel[i];

	}

	//��Myo�ṩ�µĶ�λ����ʱ������Ԫ����ʾ��������
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

	//��Myo��⵽�û������ı�ʱ��������
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

	//��Myo��⵽���ֱ���ʱ��������
	void onArmSync(myo::Myo* myo, uint64_t timestamp, myo::Arm arm,
		myo::XDirection xDirection, float rotation,
		myo::WarmupState warmupState) {
		onArm = true;
		whichArm = arm;
	}

	//��Myo���ֱ����ƿ������ƶ�ʱ��������
	void onArmUnsync(myo::Myo* myo, uint64_t timestamp) {
		onArm = false;
	}

	//��Myo����ʱ��������
	void onUnlock(myo::Myo* myo, uint64_t timestamp) {
		isUnlocked = true;
	}

	//��Myo����ʱ��������
	void onLock(myo::Myo* myo, uint64_t timestamp) {
		isUnlocked = false;
	}

	//�������ݴ洢��������
	void recData() {
		for (size_t i = 0; i < 8; i++)
			emgData[i].push_back(emgSamples[i]);
		for (size_t i = 0; i < 3; i++) {
			accelData[i].push_back(accelSamples[i]);
			orientData[i].push_back(orientSamples[i]);
		}
	}

	//ƽ���˲�������ݷ��͵��ļ�
	void filterDataToFile(string filename, UINT gestureLabel) {
		ofstream outfile(filename, ios::app);
		outfile << gestureLabel << endl;
		for (size_t i = 0; i < accelData[0].size(); i++) {
			VectorDouble temp;
			for (size_t j = 0; j < 8; j++)
				temp.push_back(emgData[j][i]);
			for (size_t j = 0; j < 3; j++)
				temp.push_back(accelData[j][i]);
			for (size_t j = 0; j < 3; j++)
				temp.push_back(orientData[j][i]);
			VectorDouble filteredValue = maf.filter(temp);
			for (size_t j = 0; j < filteredValue.size(); j++) {
				outfile << filteredValue[j] << ",";
			}
			outfile << endl;
		}
		outfile << endl;
		outfile.close();
	}

	//fft�����ݷ����ļ�
	void fftDataToFile(string filename, UINT gestureLabel) {
		ofstream outfile(filename, ios::app);
		for (size_t i = 0; i < accelData[0].size(); i++) {
			VectorDouble temp;
			for (size_t j = 0; j < 8; j++)
				temp.push_back(emgData[j][i]);
			for (size_t j = 0; j < 3; j++)
				temp.push_back(accelData[j][i]);
			for (size_t j = 0; j < 3; j++)
				temp.push_back(orientData[j][i]);
			fft.update(temp);
		}
		vector<FastFourierTransform> fftResults = fft.getFFTResults();
		outfile << gestureLabel << endl;
		for (size_t i = 0; i < 14; i++) {
			vector<double> magnitudeData = fftResults[i].getMagnitudeData();
			for (UINT m = 0; m < magnitudeData.size(); m++)
				outfile << magnitudeData[m] << ",";
			outfile << endl;
		}
		outfile << endl;
		outfile.close();
	}

	//���ǩ����
	UINT classLabelFile(string gestureName) {
		UINT gestureLabel = 1;
		UINT num;
		string tmp;
		map<string, UINT> gesture;

		//����ClassIDFile.txt�£��Ƿ���gesturename
		fstream f("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::in);
		while (f.peek() != EOF) {
			f >> tmp;
			f >> num;
			gesture[tmp] = num;
		}
		f.close();

		//��ÿ�����Ʊ�ţ���1��ʼ��
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

	//��ԭʼ���ݷ��͵��ļ��У�CSV�ļ������ļ�����ֱ�ӱ�������ʹ�ã�
	int dataToFile(string filename, UINT gestureLabel) {
		ofstream outfile(filename, ios::app);
		cerr << "Opened file" << endl;
		for (size_t i = 0; i < accelData[0].size(); i++) {
			outfile << gestureLabel << ",";
			for (int j = 0; j < 7; j++) {
				outfile << emgData[j][i] << ",";
			}
			/*for (int j = 0; j < 3; j++) {
				outfile << accelData[j][i] << ",";
			}
			for (int j = 0; j < 2; j++) {
				outfile << orientData[j][i] << ",";
			}
			outfile << orientData[2][i] << endl;*/
			outfile << emgData[7][i] << endl;
		}
		cerr << "Done writing" << endl;
		outfile.close();
		cerr << "Closing outfile" << endl;
		return accelData[0].size();
	}

	//��յ�ǰ���ݴ洢����
	void clearData() {
		for (int i = 0; i < 8; i++)
			emgData[i].clear();
		for (int i = 0; i < 3; i++) {
			accelData[i].clear();
			orientData[i].clear();
		}
	}

	//��ʼѵ��
	void startTraining(string trainingDataFile, string trainingModelFile) {
		if (!trainingData.loadDatasetFromCSVFile(trainingDataFile)) {
			cout << "ERROR: Failed to load file to dataset!" << endl;
			exit(EXIT_FAILURE);
		}
		cout << "getNumDimensions : " << trainingData.getNumDimensions() << endl;
		testData = trainingData.split(80);
		svm.getScalingEnabled();

		cout << "Ready to train data!" << endl;
		//pipeline.setPreProcessingModule(maf);
		//pipeline.setPreProcessingModule(lpf); 
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

	void getClassLabel(string classLabelFile) {
		ifstream fin(classLabelFile);
		UINT i;
		string tmp;
		while (fin >> tmp) {
			fin >> i;
			gestureNames[i] = tmp;
		}
		svm.getScalingEnabled();
		if (!pipeline.load("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\SVMModel.txt")) {
			cerr << "Filed to load the classfier model!" << endl;
			exit(EXIT_FAILURE);
		}
	}

	//����ʶ��
	void gestureRecognition() {
		VectorDouble temp;
		for (size_t i = 0; i < 8; i++)
			temp.push_back(emgSamples[i]);
		/*for (size_t i = 0; i < 3; i++)
			temp.push_back(accelSamples[i]);
		for (size_t i = 0; i < 3; i++)
			temp.push_back(orientSamples[i]);*/
		if (!pipeline.predict(temp)) {
			cerr << "Filed to perform prediction!" << endl;
			cin.ignore();
			exit(EXIT_FAILURE);
		}
		if (GetAsyncKeyState(VK_SPACE)) {
			UINT predictClassLabel = pipeline.getPredictedClassLabel();
			double maximumLikelihood = pipeline.getMaximumLikelihood();
			if (predictClassLabel)
				cout << maximumLikelihood << " " << gestureNames[predictClassLabel] << endl;
		}
	}

	//��ӡEMG���ݺ�pitch,roll,yaw
	void print() {
		cout << '\r';
		//EMG����
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


	//�洢����
	vector<vector<int>> emgData;
	vector<vector<double>> accelData;
	vector<vector<int>> orientData;

	//MYO״̬
	bool onArm;
	bool isUnlocked;
	myo::Arm whichArm;
	myo::Pose currentPose;

	//��������
	array<int, 8> emgSamples;
	array<double, 3> accelSamples;
	array<int, 3> orientSamples;

	//����ѵ��������ʶ��
	GestureRecognitionPipeline pipeline;
	double accuracy;
	SVM svm;
	ClassificationData trainingData;
	ClassificationData testData;
	map<UINT, string> gestureNames;

	//����
	FFT fft;
	MovingAverageFilter filter;
	LowPassFilter lpf;
	MovingAverageFilter maf;
};

int main(int argc, const char* argv[]) {
	while (1) {
		cout << "Function Menu" << endl;
		cout << "1.CollectData" << endl << "2.TrainModel" << endl << "3.GestureRecognition" << endl;
		char ans;
		string gesturename;
		UINT gestureLabel;
		int size;
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
				cout << "Enter name of gesture: ";
				cin >> gesturename;
				if (gesturename.size() == 0) return 0;
				cout << "Press h when ready to start recording." << endl;
				while ('h' != _getch()) {
				}
				cout << "Recording gesture. Press Esc to end recording." << endl;
				while (1) {
					hub.run(2);
					if (!collector.onArm) {
						cout << "!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!" << endl;
						return 1;
					}
					collector.print();
					collector.recData();
					if (GetAsyncKeyState(VK_ESCAPE))
						break;
				}
				cout << "Use recording? (y/n)" << endl;
				char ch;
				while ((ch = _getch()) != 'n' && ch != 'y');
				if (ch == 'y') {
					gestureLabel = collector.classLabelFile(gesturename);
					size = collector.dataToFile("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\trainingData.csv", gestureLabel);
					//collector.fftDataToFile("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\fftData.csv", gestureLabel);
					//collector.filterDataToFile("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\filterData.csv", gestureLabel);
					cout << "The size of data is " << size << endl;
					cout << "Written trainingData.csv ok!" << endl;
				}
				collector.clearData();
			}
			catch (const std::exception& e) {
				cerr << "Error: " << e.what() << std::endl;
				cerr << "Press enter to continue.";
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
				collector.getClassLabel("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt");
				while (1) {
					hub.run(50);
					if (!collector.onArm) {
						cout << "!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!!!!!" << endl;
						return 1;
					}
					collector.gestureRecognition();
					if (GetAsyncKeyState(VK_ESCAPE))
						break;
				}
			}
			catch (const std::exception& e) {
				cerr << "Error: " << e.what() << std::endl;
				cerr << "Press enter to continue.";
				break;
			}
			break;
		default:
			break;
		}
	}
}