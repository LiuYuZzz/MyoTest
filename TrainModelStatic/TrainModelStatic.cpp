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
	ClassificationData trainingData;
	trainingData.setNumDimensions(8);//8个EMG数据测试
	trainingData.setDatasetName("DummyData");
	trainingData.setInfoText("This data contains some dummy timeseries data");
	UINT currLable;
	SVM svm(SVM::LINEAR_KERNEL);
	ClassificationData testData;

	GetFilesInDirectory(gestures, "C:\\Users\\LiuYu\\Desktop\\Data\\RawData");
	system("md C:\\Users\\LiuYu\\Desktop\\Data\\Model");
	for (size_t i = 0; i < gestures.size(); i++) {
		ifstream fin(gestures[i]);
		fin >> currLable;
		size_t n;
		fin >> n;
		VectorDouble currVec(trainingData.getNumDimensions());
		vector<vector<double > >  data(n, vector<double>(14, 0));
		string tmp;
		for (size_t j = 0; j < n; j++) {
			for (size_t k = 0; k < 14; k++) {
				fin >> data[j][k];
			}
		}
		fin.close();
		for (size_t j = 0; j < n; j++) {
			for (size_t k = 0; k < currVec.size(); k++) {
				currVec[k] = data[j][k];
			}
			trainingData.addSample(currLable, currVec);
			if (!trainingData.save("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\TrainingData.csv")) {
				cout << "ERROR: Failed to save dataset to file!\n";
				return EXIT_FAILURE;
			}
		}
	}

	if (!trainingData.loadDatasetFromCSVFile("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\TrainingData.csv")) {
		cout << "ERROR: Failed to save dataset to file!" << endl;
		return EXIT_FAILURE;
	}
	cout << "Ready to train data!" << endl;
	testData = trainingData.split(80);
	svm.getScalingEnabled();
	if (!svm.train(trainingData)) {
		cout << "Filed to train classfier!" << endl;
		return EXIT_FAILURE;
	}
	if (!svm.save("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\SVMModel.txt")) {
		cout << "Failed to save the classifier model!" << endl;
		return EXIT_FAILURE;
	}
	if (!svm.load("C:\\Users\\LiuYu\\Desktop\\Data\\Model\\SVMModel.txt")) {
		cout << "Failed to load the classifier model!\n";
		return EXIT_FAILURE;
	}


	double accuracy = 0;
	for (UINT i = 0; i < testData.getNumSamples(); i++) {
		//Get the i'th test sample - this is a timeseries
		UINT classLabel = testData[i].getClassLabel();
		VectorDouble inputVector = testData[i].getSample();

		//Perform a prediction using the classifier
		if (!svm.predict(inputVector)) {
			cout << "Failed to perform prediction for test sampel: " << i << "\n";
			return EXIT_FAILURE;
		}

		//Get the predicted class label
		UINT predictedClassLabel = svm.getPredictedClassLabel();
		double maximumLikelihood = svm.getMaximumLikelihood();
		VectorDouble classLikelihoods = svm.getClassLikelihoods();
		VectorDouble classDistances = svm.getClassDistances();

		//Update the accuracy
		if (classLabel == predictedClassLabel) accuracy++;

		cout << "TestSample: " << i << "\tClassLabel: " << classLabel << "\tPredictedClassLabel: " << predictedClassLabel << "\tMaximumLikelihood: " << maximumLikelihood << endl;
	}

	cout << "Test Accuracy: " << accuracy / double(testData.getNumSamples())*100.0 << "%" << endl;
	getchar();
	return EXIT_SUCCESS;
}