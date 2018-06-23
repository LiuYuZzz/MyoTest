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

//文件是否存在
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


//数据采集
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

	//重载虚函数
	//当Myo没有配对时，被调用
	void onUnpair(myo::Myo* myo, uint64_t timestamp) {
		emgSamples.fill(0);
		ax = ay = az = 0;
		roll_w = 0;
		pitch_w = 0;
		yaw_w = 0;
		onArm = false;
		isUnlocked = false;
	}

	//当Myo提供新的EMG数据时，被调用
	void onEmgData(myo::Myo* myo, uint64_t timestamp, const int8_t* emg) {
		for (int i = 0; i < 8; i++)
		{
			emgSamples[i] = emg[i];
		}
	}

	//当Myo提供新的加速度计数据时，单位G，被调用
	void onAccelerometerData(myo::Myo* myo, uint64_t timestamp, const myo::Vector3<float>& accel) {
		ax = accel[0];
		ay = accel[1];
		az = accel[2];
	}

	//当Myo提供新的定位数据时，以四元数表示，被调用
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

	//清空当前数据存储向量
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

		//查找ClassIDFile.txt下，是否含有gesturename
		fstream f("C:\\Users\\LiuYu\\Desktop\\Data\\ClassLabelFile.txt", ios::in);
		while (f.peek() != EOF) {
			f >> tmp;
			f >> n;
			gesturenames[tmp] = n;
		}
		f.close();

		//给每个手势编号（从1开始）
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

	//将原始数据发送到文件中
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
					//生成ClassID.txt文件
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