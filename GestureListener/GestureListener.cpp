#include <array>

#include <myo/myo.hpp>
#include <GRT/GRT.h>

using namespace std;
using namespace GRT;

class GestureListener : public myo::DeviceListener {
public:
	GestureListener(string classIDFile, string dataModelFile)
		:emgSamples(), onArm(false), isUnlocked(false), roll_w(0), pitch_w(0), yaw_w(0), currentPose() {
		emgSamples.fill(0);
		roll = 0;
		pitch = 0;
		yaw = 0;
		ax = ay = az = 0;

		ifstream fin(classIDFile);
		UINT i;
		string tmp;
		while (fin >> tmp) {
			fin >> i;
			gesturenames[i] = tmp;
		}

		dtw.enableNullRejection(true);
		dtw.setNullRejectionCoeff(8);
		dtw.enableTrimTrainingData(true, 0.1, 90);
		if (!dtw.load(dataModelFile)) {
			cerr << "Filed to load the classfier model!" << endl;
			exit(EXIT_FAILURE);
		}
		cerr << "Device listener constructed!" << endl;
	}

	void onGesture(double confidence, string gesturename) {
		cout << confidence << " " << gesturename << endl;
	}

	void recData() {
		vector<double> temp;
		for (int i = 0; i < 8; i++) {
			temp.push_back(emgSamples[i]);
		}
		temp.push_back(ax);
		temp.push_back(ay);
		temp.push_back(az);
		temp.push_back(roll_w);
		temp.push_back(pitch_w);
		temp.push_back(yaw_w);
		data.push_back(temp);

		//前100个数据不要
		int buffersize = 100;
		if (data.size() > buffersize + 1) {
			MatrixDouble window;
			for (int i = 0; i < buffersize; i++) {
				VectorDouble currVec;
				//测试EMG数据
				for (int j = 0; j < 8; j++) {
					currVec.push_back(data[data.size() - buffersize + i][j]);
				}
				window.push_back(currVec);
			}
			if (!dtw.predict(window)) {
				cerr << "Filed to perform prediction!" << endl;
				exit(EXIT_FAILURE);
			}
			UINT predictClassLabel = dtw.getPredictedClassLabel();
			double maximumLikelihood = dtw.getMaximumLikelihood();
			if (predictClassLabel)
				onGesture(maximumLikelihood, gesturenames[predictClassLabel]);
		}
	}

	//清空data
	void clearData() {
		for (int i = 0; i < 8; i++) {
			data[i].clear();
		}
	}

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

	DTW dtw;
	vector<vector<double> > data;
	map<UINT, string> gesturenames;
	array<double, 8> emgSamples;
	bool onArm;
	bool isUnlocked;
	int roll_w, pitch_w, yaw_w;
	float roll, pitch, yaw, ax, ay, az;
	myo::Arm whichArm;
	myo::Pose currentPose;
};

int main(int argc, char** argv) {
	try {
		myo::Hub hub("com.example.gesturemyo");
		cout << "Attempting to find a Myo ..." << endl;
		myo::Myo* myo = hub.waitForMyo(10000);
		if (!myo)
			throw runtime_error("Unable to find a Myo!");
		cout << "Connected to a Myo armband!" << endl << endl;
		myo->setStreamEmg(myo::Myo::streamEmgEnabled);
		GestureListener collector("C:\\Users\\LiuYu\\Desktop\\Data\\ClassIDFile.txt",
			"C:\\Users\\LiuYu\\Desktop\\Data\\Model\\DTWModel.txt");
		hub.addListener(&collector);
		while (1) {
			hub.run(10);
			if (!collector.onArm) {
				cout << "!!!!!!!!!!!!!!Please sync the Myo!!!!!!!!!!!!!!!!!!!!!" << endl;
				return 1;
			}
			collector.recData();
			if (GetAsyncKeyState(VK_ESCAPE))
				break;
		}
	}
	catch(const std::exception& e){
		cerr << "Error: " << e.what() << std::endl;
		cerr << "Press enter to continue.";
		cin.ignore();
		return 1;
	}
}