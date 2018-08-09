#include "slamBase.h"

FRAME readFrame(int index, ParameterReader& pd);
double normofTransform(Mat rvec,Mat tvec);

int visualOdometry()
{
    ParameterReader pd;
    PointCloud::Ptr cloud(new PointCloud);
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());
    cout<<"Initializing ..."<<endl;
    int currIndex=startIndex;

    FRAME lastFrame;//=readFrame(currIndex,pd);
    string json_path,json_index,json_head;
    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    json_index=std::to_string(startIndex);
    json_head=pd.getData("json_path");
    json_path=json_head+json_index;
    CAMERA_INTRINSIC_PARAMETERS camera=getCamera(pd);
    image2PointCloud(json_path,lastFrame);
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    pcl::visualization::CloudViewer viewer("viewer");

    bool visualize = pd.getData("visualize_pointcloud")==string("yes");
    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );
    
    for(currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
	cout<<"Reading files"<<currIndex<<endl;
	json_index=std::to_string(currIndex);
        json_path=json_head+json_index;
	FRAME currFrame;//readFrame(currIndex,pd);
        image2PointCloud(json_path,currFrame);
	computeKeyPointsAndDesp(currFrame,detector,descriptor);
	RESULT_OF_PNP result=estimateMotion(lastFrame,currFrame,camera);
	if(result.inliers<min_inliers)
	    continue;
	double norm=normofTransform(result.Rvec,result.Tvec);
	cout<<"norm="<<norm<<endl;
	if(norm>=max_norm)
	    continue;
	Eigen::Isometry3d T=cvMat2Eigen(result.Rvec,result.Tvec);
	cout<<"T="<<T.matrix()<<endl;
	joinPointCloud(lastFrame,currFrame,T,camera);
        cloud=currFrame.cloud;
	if(visualize==true)
	    viewer.showCloud(cloud);
	lastFrame=currFrame;
    }
    pcl::io::savePCDFile("/home/wayne/Projects/RGBD_SLAM/pointcloud/result.pcd",*cloud);
    return 0;
}

FRAME readFrame(int index,ParameterReader& pd)
{
    FRAME f;
    string rgbDir=pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");
    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");
    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb=imread(filename);
    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;
    f.img_3d=imread(filename,-1);
    return f;
}

double normofTransform(Mat rvec,Mat tvec)
{
    return fabs(min(norm(rvec),2*M_PI-norm(rvec)))+fabs(norm(tvec));
}

