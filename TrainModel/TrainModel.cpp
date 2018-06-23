#include <GRT/GRT.h>
#include <Windows.h>
#include <random>
#define WINDOWS

using namespace std;
using namespace GRT;

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

void GetFilesInDirectory(std::vector<string> &out, const string &directory)
{
#ifdef WINDOWS
	HANDLE dir;
	WIN32_FIND_DATA file_data;

	if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
		return; /* No files found */

	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = directory + "/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	} while (FindNextFile(dir, &file_data));

	FindClose(dir);
#else
	DIR *dir;
	class dirent *ent;
	class stat st;

	dir = opendir(directory);
	while ((ent = readdir(dir)) != NULL) {
		const string file_name = ent->d_name;
		const string full_file_name = directory + "/" + file_name;

		if (file_name[0] == '.')
			continue;

		if (stat(full_file_name.c_str(), &st) == -1)
			continue;

		const bool is_directory = (st.st_mode & S_IFDIR) != 0;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	}
	closedir(dir);
#endif
} // GetFilesInDirectory


int main(int argc, const char* argv[]) {
	vector<string> gestures(0, "");
	TimeSeriesClassificationData trainingData;
	trainingData.setNumDimensions(8);//8个EMG数据测试
	trainingData.setDatasetName("DummyData");
	trainingData.setInfoText("This data contains some dummy timeseries data");
	UINT currLable;
	DTW dtw;
	TimeSeriesClassificationData testData;

	GetFilesInDirectory(gestures, "C:\\Users\\LiuYu\\Desktop\\Data\\RawData");
	system("md C:\\Users\\LiuYu\\Desktop\\Data\\Model");
	for (size_t i = 0; i < gestures.size(); i++) {
		ifstream fin(gestures[i]);
		fin >> currLable;
		int n;
		fin >> n;
		MatrixDouble trainingSample;
		VectorDouble currVec(trainingData.getNumDimensions());
		vector<vector<double > >  data(n,vector<double>(14,0));
		string tmp;
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < 14; k++) {
				fin >> data[j][k];
			}
		}
		fin.close();
		for (size_t j = 0; j < n; j++) {
			for (UINT k = 0; k < currVec.size(); k++) {
				currVec[k] = data[j][k];
			}
			trainingSample.push_back(currVec);
		}
		trainingData.addSample(currLable, trainingSample);
	}


	if (!trainingData.save("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\TrainingData.grt")) {
		cout << "ERROR: Failed to save dataset to file!\n";
		return EXIT_FAILURE;
	}
	if (!trainingData.load("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\TrainingData.grt")) {
		cout << "ERROR: Failed to load dataset from file!\n";
		return EXIT_FAILURE;
	}
	
	testData = trainingData.split(80);
	dtw.enableNullRejection(true);
	dtw.setNullRejectionCoeff(8);
	dtw.enableTrimTrainingData(true, 0.1, 90);
	if (!dtw.train(trainingData)) {
		cout << "Filed to train classfier!" << endl;
		return EXIT_FAILURE;
	}
	if (!dtw.save("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\DTWModel.txt")) {
		cout << "Failed to save the classifier model!" << endl;
		return EXIT_FAILURE;
	}
	if (!dtw.load("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\DTWModel.txt")) {
		cout << "Failed to load the classifier model!\n";
		return EXIT_FAILURE;
	}


	double accuracy = 0;
	for (UINT i = 0; i<testData.getNumSamples(); i++) {
		//Get the i'th test sample - this is a timeseries
		UINT classLabel = testData[i].getClassLabel();
		MatrixDouble timeseries = testData[i].getData();

		//Perform a prediction using the classifier
		if (!dtw.predict(timeseries)) {
			cout << "Failed to perform prediction for test sampel: " << i << "\n";
			return EXIT_FAILURE;
		}

		//Get the predicted class label
		UINT predictedClassLabel = dtw.getPredictedClassLabel();
		double maximumLikelihood = dtw.getMaximumLikelihood();
		VectorDouble classLikelihoods = dtw.getClassLikelihoods();
		VectorDouble classDistances = dtw.getClassDistances();

		//Update the accuracy
		if (classLabel == predictedClassLabel) accuracy++;

		cout << "TestSample: " << i << "\tClassLabel: " << classLabel << "\tPredictedClassLabel: " << predictedClassLabel << "\tMaximumLikelihood: " << maximumLikelihood << endl;
	}

	cout << "Test Accuracy: " << accuracy / double(testData.getNumSamples())*100.0 << "%" << endl;
	getchar();
	return EXIT_SUCCESS;
}