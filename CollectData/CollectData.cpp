#define _USE_MATH_DEFINES
#include <Windows.h> 
#include <fstream>
#include <sstream>
#include <conio.h>
#include <iostream>
#include <array>
#include <cmath>
#include <string>
#include <map>
//#define WINDOWS
#define NUMPARAM 13

#include <myo/myo.hpp>

using namespace std;

//�ļ��Ƿ����
inline bool file_exists(string name) {
	ifstream f(name.c_str());
	if (f.good()) {
		f.close();
		return true;
	}
	else {
		f.close();
		return false;
	}
}

inline bool label_exists(string name) {

}


//���ݲɼ�
class DataCollector : public myo::DeviceListener {
public:
	DataCollector()
		:emgSamples(), onArm(false), isUnlocked(false), roll_w(0), pitch_w(0), yaw_w(0), currentPose()
	{
		emgdata = vector<vector<int8_t>>(8, vector<int8_t>(0, 0));
		accel = vector<vector<double>>(3, vector<double>(0, 0));
		orient = vector<vector<double>>(3, vector<double>(0, 0));
		emgSamples.fill(0);
		roll = 0;
		pitch = 0;
		yaw = 0;
		ax = ay = az = 0;
	}

	//�����麯��
	//��Myoû�����ʱ��������
	void onUnpair(myo::Myo* myo, uint64_t timestamp) {
		emgSamples.fill(0);
		ax = ay = az = 0;
		roll_w = 0;
		pitch_w = 0;
		yaw_w = 0;
		onArm = false;
		isUnlocked = false;
	}

	//��Myo�ṩ�µ�EMG����ʱ��������
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg) {
		for (int i = 0; i < 8; i++)
		{
			emgSamples[i] = emg[i];
		}
	}

	//��Myo�ṩ�µļ��ٶȼ�����ʱ����λG��������
	void onAccelerometerData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float>& accel) {
		ax = accel[0];
		ay = accel[1];
		az = accel[2];
	}

	//��Myo�ṩ�µĶ�λ����ʱ������Ԫ����ʾ��������
	void onOrientationData(myo::Myo* myo, uint64_t timestamp,
		const myo::Quaternion<float>& rotation) {
		roll = atan2(2.0f * (rotation.w() * rotation.x() + rotation.y() * rotation.z()),
			1.0f - 2.0f * (rotation.x() * rotation.x() + rotation.y() * rotation.y()));
		pitch = asin(max(-1.0f, min(1.0f, 2.0f * (rotation.w() * rotation.y() - rotation.z() * rotation.x()))));
		yaw = atan2(2.0f * (rotation.w() * rotation.z() + rotation.x() * rotation.y()),
			1.0f - 2.0f * (rotation.y() * rotation.y() + rotation.z() * rotation.z()));
		roll_w = static_cast<int>((roll + (float)M_PI) / (M_PI * 2.0f) * 18);
		pitch_w = static_cast<int>((pitch + (float)M_PI / 2.0f) / M_PI * 18);
		yaw_w = static_cast<int>((yaw + (float)M_PI) / (M_PI * 2.0f) * 18);
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
		for (int i = 0; i < 8; i++) {
			emgdata[i].push_back(emgSamples[i]);
		}
		accel[0].push_back(ax);
		accel[1].push_back(ay);
		accel[2].push_back(az);
		orient[0].push_back(roll_w);
		orient[1].push_back(pitch_w);
		orient[2].push_back(yaw_w);
	}

	//��յ�ǰ���ݴ洢����
	void clearData() {
		for (int i = 0; i < 8; i++)
			emgdata[i].clear();
		for (int i = 0; i < 3; i++) {
			accel[i].clear();
			orient[i].clear();
		}
	}

	UINT classLabelFile(string gesturename) {
		UINT currLabel = 1;
		UINT n;
		string tmp;
		map<string, UINT> gesturenames;

		//����ClassIDFile.txt�£��Ƿ���gesturename
		fstream f("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::in);
		while (f.peek() != EOF) {
			f >> tmp;
			f >> n;
			gesturenames[tmp] = n;
		}
		f.close();

		//��ÿ�����Ʊ�ţ���1��ʼ��
		if (gesturenames.size() == 0) {
			currLabel = 1;
			gesturenames[gesturename] = currLabel;
			f.open("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::app);
			f << gesturename << endl;
			f << gesturenames[gesturename] << endl;
			f.close();
			return currLabel;	
		}
		if (gesturenames.count(gesturename)) {
			currLabel = gesturenames[gesturename];
			return currLabel;
		}
		else {
			currLabel = gesturenames.size() + 1;
			gesturenames[gesturename] = currLabel;
			f.open("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::app);
			f << gesturename << endl;
			f << gesturenames[gesturename] << endl;
			f.close();
			return currLabel;
		}
	}

	//��ԭʼ���ݷ��͵��ļ���
	void dataToFile(string filename,UINT label) {
		ofstream outfile(filename);
		cerr << "Opened file" << endl;
		outfile << label << endl;
		outfile << accel[0].size() << endl;
		for (size_t i = 0; i < accel[0].size(); i++) {
			for (int j = 0; j < 8; j++) {
				outfile << static_cast<int>(emgdata[j][i]) << " ";
			}
			for (int j = 0; j < 3; j++) {
				outfile << accel[j][i] << " ";
			}
			for (int j = 0; j < 3; j++) {
				outfile << orient[j][i] << " ";
			}
			outfile << endl;
		}
		cerr << "Done writing" << endl;
		outfile.close();
		cerr << "Closing outfile" << endl;
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
		cout << '[' << string(roll_w, '*') << string(18 - roll_w, ' ') << ']'
			<< '[' << string(pitch_w, '*') << string(18 - pitch_w, ' ') << ']'
			<< '[' << string(yaw_w, '*') << string(18 - yaw_w, ' ') << ']';
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

	array<int8_t, 8> emgSamples;
	bool onArm;
	bool isUnlocked;
	int roll_w, pitch_w, yaw_w;
	float roll, pitch, yaw, ax, ay, az;
	vector<vector<int8_t>> emgdata;
	vector<vector<double>> accel;
	vector<vector<double>> orient;
	myo::Arm whichArm;
	myo::Pose currentPose;
};

int main(int argc, const char* argv[]) {
	system("md C:\\Users\\LiuYu\\Desktop\\Data\\RawData");
	try {
		myo::Hub hub("com.example.gesturemyo");
		cout << "Attempting to find a Myo ..." << endl;
		myo::Myo* myo = hub.waitForMyo(10000);
		if (!myo) {
			throw runtime_error("Unable to find a Myo!");
		}
		cout << "Connected to a Myo armband!" << endl << endl;
		myo->setStreamEmg(myo::Myo::streamEmgEnabled);
		DataCollector collector;
		hub.addListener(&collector);
		map<string, UINT> gesturenames;
		while (1) {
			cout << "Enter name of gesture: ";
			string gesturename;
			cin >> gesturename;
			if (gesturename.size() == 0) return 0;
			cout << "And number of times you'll perform it: ";
			int t = 0;
			cin >> t;
			int fno = 0;
			UINT label;
			while (t--) {
				while (file_exists("C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\" + gesturename + "_" + to_string(fno) + ".txt")) fno++;
				string filename = "C:\\Users\\LiuYu\\Desktop\\Data\\RawData\\" + gesturename + "_" + to_string(fno) + ".txt";
				cout << "Press h when ready to start recording." << endl;
				while ('h' != _getch()) {
				}
				cout << "Recording gesture. Press Esc to end recording." << endl;
				while (1) {
					hub.run(500);
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
				char ans;
				while ((ans = _getch()) != 'n' && ans != 'y');
				if (ans == 'y') {
					//����ClassID.txt�ļ�
					label = collector.classLabelFile(gesturename);
					collector.dataToFile(filename,label);
					cout << "Written as " << filename << endl;
				}
				collector.clearData();
			}
		}
	}
	catch (const std::exception& e) {
		cerr << "Error: " << e.what() << std::endl;
		cerr << "Press enter to continue.";
		cin.ignore();
		return 1;
	}
}