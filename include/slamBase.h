#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//using namespace cv;

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

#include <Python.h>

struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx,cy,fx,fy;
};

//void image2PointCloud(string json_path);

struct FRAME
{
    cv::Mat rgb;
    cv::Mat img_3d;
    PointCloud::Ptr cloud;//;(new PointCloud);
    cv::Mat desp;
    cv::Mat mask;
    vector<cv::KeyPoint>kp;
};

struct RESULT_OF_PNP
{
    cv::Mat Rvec,Tvec;
    int inliers;
};



class ParameterReader
{
public:
    ParameterReader(string filename="/home/wayne/Projects/RGBD_SLAM/src/parameters.txt")
    {
        
	ifstream fin(filename.c_str());
	if(!fin)
	{
	    cerr<<"parameter file does not exist."<<endl;
	    return;
	}
	while(!fin.eof())
        {
            string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr( 0, pos );
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if ( !fin.good() )
                break;
        }
    }
    string getData( string key )
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    map<string, string> data;
};

void computeKeyPointsAndDesp(FRAME& frame, string detector, string descriptor);
//PointCloud::Ptr image2PointCloud(string json_path);

RESULT_OF_PNP estimateMotion(FRAME& frame1,FRAME& frame2,CAMERA_INTRINSIC_PARAMETERS& camera);

void stereo_disp(cv::Mat img1,cv::Mat img2,cv::Mat map_x1,cv::Mat map_y1, cv::Mat map_x2,cv::Mat map_y2,FRAME* frame,string i);

FRAME image2PointCloud(string json_path,double count,float pc_thresh);

Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME newFrame,Eigen::Isometry3d T,CAMERA_INTRINSIC_PARAMETERS& camera);

CAMERA_INTRINSIC_PARAMETERS getCamera(ParameterReader pd);

void get_Camera(ParameterReader pd,cv::Mat cam_mtx1,cv::Mat cam_mtx2,cv::Mat dist1,cv::Mat dist2,cv::Mat R1,cv::Mat P1,cv::Mat R2,cv::Mat P2,cv::Mat Q);

double normofTransform(cv::Mat rvec,cv::Mat tvec);

void visualOdometry();

