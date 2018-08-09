#include "slamBase.h"

FRAME image2PointCloud(string json_path)
{
    FRAME f;
    cv::Mat rgb;
    vector<Mat> channels;
    cv::Mat dep_x=Mat::zeros(Size(3008,4112),CV_32FC1);
    cv::Mat dep_y=Mat::zeros(Size(3008,4112),CV_32FC1);
    cv::Mat dep_z=Mat::zeros(Size(3008,4112),CV_32FC1);
    cv::Mat depth=Mat::zeros(Size(3008,4112),CV_32FC3);
    stringstream ss;
    string i;
    char tmp=json_path[40];
    ss<<tmp;
    ss>>i;
    cout<<i<<endl;
    /*    
    Py_Initialize();
    PyRun_SimpleString("import sys\nsys.path.append('/home/wayne/Projects/RGBD_SLAM/src/')");
    PyObject* myModuleString=PyString_FromString((char*)"LocalRunReconstruction");
    PyObject* myModule=PyImport_Import(myModuleString);
    if(myModule!=NULL)
    {
        PyObject* myFunction=PyObject_GetAttrString(myModule,(char*)"main");
    	PyObject* args;
        const char* cstr=json_path.c_str();	
	args=PyTuple_Pack(1,PyString_FromString(cstr));
    	PyObject* myResult=PyObject_CallObject(myFunction,args);
    }
    else 
    {
	cout<<"import failed"<<endl;
    }
    */
     
    rgb=cv::imread("/home/wayne/Projects/RGBD_SLAM/img/color"+i+".jpg",-1);
     
    FileStorage fx("/home/wayne/Projects/RGBD_SLAM/img/x_3d"+i+".xml",FileStorage::READ);
    fx["dep_x"]>>dep_x;
    FileStorage fy("/home/wayne/Projects/RGBD_SLAM/img/y_3d"+i+".xml",FileStorage::READ);
    fy["dep_y"]>>dep_y;
    FileStorage fz("/home/wayne/Projects/RGBD_SLAM/img/z_3d"+i+".xml",FileStorage::READ);
    fz["dep_z"]>>dep_z;
    channels.push_back(dep_x);
    channels.push_back(dep_y);
    channels.push_back(dep_z);
     
    merge(channels,depth);
    FileStorage d("/home/wayne/Projects/RGBD_SLAM/img/depth"+i+".xml",FileStorage::WRITE);
    d<<"depth"<<depth;
    f.img_3d=depth;
    f.rgb=rgb;
    
    //PointCloud::Ptr cloud(new PointCloud);
    f.cloud=PointCloud::Ptr(new PointCloud());
    for(int r=0;r<depth.rows;r++)
    {
	for(int c=0;c<depth.cols;c++)
	{
	    PointT p;
	    Vec3f pix=depth.at<Vec3f>(r,c);
	    float x=pix.val[0];
	    float y=pix.val[1];
	    float z=pix.val[2];
	    if(z<1000)// && r>=depth.rows*0.3 && r<=depth.rows*0.7)
	    {
	    	p.z=z;
	    	p.x=x;
	    	p.y=y;
	    	p.b=rgb.ptr<uchar>(r)[c*3];
	    	p.g=rgb.ptr<uchar>(r)[c*3+1];
	    	p.r=rgb.ptr<uchar>(r)[c*3+2];
	    	f.cloud->points.push_back(p);
	    }
	}
    }
     
    f.cloud->height=1;
    f.cloud->width=f.cloud->points.size();
    cout<<"point cloud size="<<f.cloud->points.size()<<endl;
    f.cloud->is_dense=false;
    
    pcl::io::savePCDFile("/home/wayne/Projects/RGBD_SLAM/pointcloud/pointcloud"+i+".pcd",*f.cloud);
    //cloud->points.clear();
    cout<<"Point cloud saved."<<endl;
    //*f.cloud=*cloud;
    return f;
}

void computeKeyPointsAndDesp(FRAME& frame, string detector,string descriptor)
{
    Ptr<FeatureDetector> detector_comp;
    Ptr<DescriptorExtractor> descriptor_comp;
    initModule_nonfree();
    detector_comp = cv::FeatureDetector::create( detector.c_str() );
    descriptor_comp = cv::DescriptorExtractor::create( descriptor.c_str() );
    if(!detector_comp || !descriptor_comp)
    {
 	cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
	return;
    }
    detector_comp->detect(frame.rgb,frame.kp);
    descriptor_comp->compute(frame.rgb,frame.kp,frame.desp);
    return;
}

RESULT_OF_PNP estimateMotion( FRAME& frame1, FRAME& frame2, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    static ParameterReader pd;
    vector<DMatch>matches;
    FlannBasedMatcher matcher;
    matcher.match(frame1.desp,frame2.desp,matches);
    cout<<"find total "<<matches.size()<<" matches."<<endl;
    vector<DMatch>goodMatches;
    double maxDis=0;
    double minDis=999999;
    double good_match_threshold = atof( pd.getData( "good_match_threshold" ).c_str() );
    for(size_t i=0;i<matches.size();i++)
    {
	if(matches[i].distance<minDis)
	    minDis=matches[i].distance;
        if(matches[i].distance>maxDis)
	    maxDis=matches[i].distance;
    }
    for(size_t i=0;i<matches.size();i++)
    {
	if(matches[i].distance<good_match_threshold*minDis)
	    goodMatches.push_back(matches[i]);
    }
    //////////////////////
    Mat img_matches;
    drawMatches(frame1.rgb,frame1.kp,frame2.rgb,frame2.kp,goodMatches,img_matches,Scalar::all(-1),Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("/home/wayne/Projects/RGBD_SLAM/img/tmp.jpg",img_matches);
    //////////////////
    cout<<"good matches: "<<goodMatches.size()<<endl;
    vector<Point3f> pts_obj;
    vector<Point2f>pts_img;
    for(size_t i=0;i<goodMatches.size();i++)
    {
	Point2f p=frame1.kp[goodMatches[i].queryIdx].pt;
 	pts_img.push_back(Point2f(frame2.kp[goodMatches[i].trainIdx].pt));
	Vec3f pix=frame1.img_3d.at<Vec3f>(p.y,p.x);   
	float x=pix.val[0];
        float y=pix.val[1];
        float z=pix.val[2];
    
	Point3f pd (x,y,z);/*frame1.img_3d.ptr<float>(int(p.y))[int(p.x)],
		    frame1.img_3d.ptr<float>(int(p.y))[int(p.x)+1],
  		    frame1.img_3d.ptr<float>(int(p.y))[int(p.x)+2]);*/
	pts_obj.push_back(pd);

    }
    double camera_matrix_data[3][3]={{camera.fx, 0, camera.cx},
				     {0, camera.fy, camera.cy},
				     {0, 0, 1}};
    cout<<"solving pnp"<<endl;
    Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    Mat rvec, tvec, inliers;
    solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
    RESULT_OF_PNP result;
    result.Rvec=rvec;
    result.Tvec=tvec;
    result.inliers=inliers.rows;
    //cout<<"R= "<<result.Rvec<<endl;
    //cout<<"T= "<<result.Tvec<<endl;
    return result;
}

Eigen::Isometry3d cvMat2Eigen(Mat& rvec, Mat& tvec)
{
    Mat R;
    Rodrigues(rvec,R);
    Eigen::Matrix3d r;
    cv2eigen(R,r);
    cout<<"R="<<r<<endl;
    Eigen::Isometry3d T=Eigen::Isometry3d::Identity();
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0),
				       tvec.at<double>(0,1),
				       tvec.at<double>(0,2));
    T=angle;
    T(1,0)=-T(1,0);
    T(1,1)=-T(1,1);
    T(1,2)=-T(1,2);
    T(2,0)=-T(2,0);
    T(2,1)=-T(2,1);
    T(2,2)=-T(2,2);
    T(0,3)=tvec.at<double>(0,0);
    T(1,3)=-tvec.at<double>(0,1);
    T(2,3)=-tvec.at<double>(0,2);
    return T;
}

PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,FRAME newFrame,Eigen::Isometry3d T,CAMERA_INTRINSIC_PARAMETERS& camera)
{
    //PointCloud::Ptr newCloud=newFrame.cloud;
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*(original),*output,T.matrix());
    
    *output+=*newFrame.cloud;
    
    //output->height=1;
    //output->width=output->points.size();
    //output->is_dense=false; 
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    /*
    double gridsize=atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize,gridsize,gridsize);
    voxel.setInputCloud(output);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    */
    double gridsize = atof( pd.getData("voxel_grid").c_str() );
    voxel.setLeafSize( gridsize, gridsize, gridsize );
    voxel.setInputCloud(output );
    PointCloud::Ptr tmp( new PointCloud() );
    voxel.filter( *tmp );
    
    return output;//tmp;
}

CAMERA_INTRINSIC_PARAMETERS getCamera(ParameterReader pd)
{
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx=std::atof((pd.getData("camera.cx")).c_str());
    camera.cy=std::atof((pd.getData("camera.cy")).c_str());
    camera.fx=std::atof((pd.getData("camera.fx")).c_str());
    camera.fy=std::atof((pd.getData("camera.fy")).c_str());
    return camera;
}


void visualOdometry()
{
    ParameterReader pd;
    PointCloud::Ptr cloud(new PointCloud());
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());
    cout<<"Initializing ..."<<endl;
    int currIndex=startIndex;

    string json_path,json_index,json_head,json_tail;
    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    stringstream convert;
    convert<<startIndex;
    json_index=convert.str();
    json_head=pd.getData("json_path");
    json_tail=".json";
    json_path=json_head+json_index+json_tail;
    
    CAMERA_INTRINSIC_PARAMETERS camera=getCamera(pd);
    cout<<camera.fx<<endl;
    FRAME lastFrame=image2PointCloud(json_path);
    cloud=lastFrame.cloud;
    
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    cout<<"success compute key."<<endl;
    pcl::visualization::CloudViewer viewer("viewer");
    
    bool visualize = pd.getData("visualize_pointcloud")==string("yes");
    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );
    if(visualize==true)
	viewer.showCloud(cloud);
    for(currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
	cout<<"Reading files"<<currIndex<<endl;
        convert.str(std::string());
 	convert<<currIndex;
	json_index=convert.str();
        json_path=json_head+json_index+json_tail;
        FRAME currFrame=image2PointCloud(json_path);
	computeKeyPointsAndDesp(currFrame,detector,descriptor);
	RESULT_OF_PNP result=estimateMotion(lastFrame,currFrame,camera);
	cout<<"inliers="<<result.inliers<<endl;
	if(result.inliers<min_inliers)
	    continue;
	double norm=normofTransform(result.Rvec,result.Tvec);
	cout<<"norm="<<norm<<endl;
	if(norm>=max_norm)
	    continue;
	Eigen::Isometry3d T=cvMat2Eigen(result.Rvec,result.Tvec);
	cout<<"T="<<T.matrix()<<endl;
	cloud=joinPointCloud(cloud,currFrame,T,camera);
	if(visualize==true)
	    viewer.showCloud(cloud);
	lastFrame=currFrame;
    }
    cout<<"done!"<<endl;
    cloud->height=1;
    cloud->width=cloud->points.size();
    cout<<"point cloud size="<<cloud->points.size()<<endl;
    cloud->is_dense=false;
    pcl::io::savePCDFile("/home/wayne/Projects/RGBD_SLAM/pointcloud/result.pcd",*cloud);
}

double normofTransform(Mat rvec,Mat tvec)
{
    return (fabs(min(norm(rvec),2*M_PI-norm(rvec)))+fabs(norm(tvec)));
}
