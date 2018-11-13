#include "slamBase.h"
using namespace cv;

void stereo_disp(cv::Mat img1,cv::Mat img2,Mat map_x1,Mat map_y1, Mat map_x2,Mat map_y2,Mat Q,FRAME* frame,string i)
{
    Mat gray1,gray2,dst1,dst2,disp,dispbm,dispbm16,mask,mask1,mask2,img_3d,disp16,tmp1,tmp2;
    ParameterReader pd;
    int disp12=atof(pd.getData("disp12Max").c_str());
    int block_size=atof(pd.getData("block_size").c_str());
    int num_disp=atof(pd.getData("num_disp").c_str());
    int tmp_len=(block_size-1)/2;
    int pre_j,left_col,tmp_disp;
    int found=1;
    float match_per,per,left_j;
    double min_v1,max_v1,min_v2,max_v2;
   
    cvtColor(img1,gray1,CV_BGR2GRAY);
    cvtColor(img2,gray2,CV_BGR2GRAY);
    remap(gray1,dst1,map_x1,map_y1,INTER_LINEAR);
    remap(gray2,dst2,map_x2,map_y2,INTER_LINEAR);
    if(i.length()==1)
    {
    	imwrite("/home/wayne/Desktop/data_0911/image_0/00000"+i+".png",dst1);
     	imwrite("/home/wayne/Desktop/data_0911/image_1/00000"+i+".png",dst2);
    }
    else if(i.length()==2)
    {
 	imwrite("/home/wayne/Desktop/data_0911/image_0/0000"+i+".png",dst1);
        imwrite("/home/wayne/Desktop/data_0911/image_1/0000"+i+".png",dst2);
    }
    else
    {
	imwrite("/home/wayne/Desktop/data_0911/image_0/000"+i+".png",dst1);
        imwrite("/home/wayne/Desktop/data_0911/image_1/000"+i+".png",dst2);
    }

    Ptr<StereoBM> bm=StereoBM::create(num_disp,block_size);
    Ptr<StereoSGBM> sgbm=StereoSGBM::create(0,num_disp,block_size,8*3*block_size*block_size,32*3*block_size*block_size,disp12);	
    sgbm->compute(dst1,dst2,disp);
    bm->compute(dst1,dst2,dispbm);
    dispbm.convertTo(dispbm16,CV_32FC1);
    disp.convertTo(disp16,CV_32FC1);
    disp16=disp16/16.0; 
    dispbm16=dispbm16/16.0;
    mask1=Mat::zeros(480,752,CV_8UC1);
    mask2=Mat::zeros(480,752,CV_8UC1);
    mask=Mat::zeros(480,752,CV_32FC1);
    minMaxLoc(disp16,&min_v1,&max_v1);
    minMaxLoc(dispbm16,&min_v2,&max_v2);
    if(min_v1>0)
	tmp1=disp16>min_v1;
    else tmp1=disp16>0;
    tmp1.convertTo(mask1,CV_8UC1);
    if(min_v2>0)
	tmp2=dispbm16>min_v2;
    else tmp2=dispbm16>0;
    tmp2.convertTo(mask2,CV_8UC1);
    
    for(int r=0;r<mask1.rows;r++)
    {
	for(int c=0;c<mask.cols;c++)
	{
	    if(abs(disp16.at<float>(r,c)-dispbm16.at<float>(r,c))>0.1 || mask1.at<short>(r,c)==0 || mask2.at<short>(r,c)==0)
		mask.at<float>(r,c)=0;
	    else mask.at<float>(r,c)=1;
	}
    }   
    reprojectImageTo3D(dispbm16,img_3d,Q);
    frame->img_3d=img_3d.clone();
    frame->rgb=dst1.clone();
    frame->mask=mask.clone();
}

FRAME image2PointCloud(string path,double count,float pc_thresh,Mat map_x1,Mat map_y1,Mat map_x2,Mat map_y2,Mat Q)
{
    FRAME f;
    stringstream ss;
    string i;
    ParameterReader pd;
    float d;
    
    ss<<count;
    i=ss.str();
    string data_set=pd.getData("data");
    string left_p=path+"left/"+data_set+"left/"+i+".jpg";
    string right_p=path+"right/"+data_set+"right/"+i+".jpg";
    Mat img1=imread(left_p);
    Mat img2=imread(right_p);
    stereo_disp(img1,img2,map_x1,map_y1,map_x2,map_y2,Q,&f,i);
    
    
    f.cloud=PointCloud::Ptr(new PointCloud());
    for(int r=0;r<f.img_3d.rows;r++)
    {
	for(int c=0;c<f.img_3d.cols;c++)
	{
	    PointT p;
	    Vec3f pix=f.img_3d.at<Vec3f>(r,c);
	    float x=pix.val[0];
	    float y=pix.val[1];
	    float z=pix.val[2];
    	    d=sqrt(x*x+y*y+z*z);
	    if(d<pc_thresh && f.mask.at<float>(r,c)==1)
	    {
	    	p.z=z;
	    	p.x=x;
	    	p.y=y;
	    	p.b=f.rgb.ptr<uchar>(r)[c*3];
	    	p.g=f.rgb.ptr<uchar>(r)[c*3+1];
	    	p.r=f.rgb.ptr<uchar>(r)[c*3+2];
	    	f.cloud->points.push_back(p);
		
	    }
	    //else f.mask.at<float>(r,c)=0;
	
	}
    }
   
    f.cloud->height=1;
    f.cloud->width=f.cloud->points.size();
    //cout<<"point cloud size="<<f.cloud->points.size()<<endl;
    f.cloud->is_dense=false;
    
    return f;
}

void computeKeyPointsAndDesp(FRAME& frame, string detector,string descriptor)
{
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

    f2d->detect(frame.rgb, frame.kp );
   
    //-- Step 2: Calculate descriptors (feature vectors)    
    f2d->compute( frame.rgb, frame.kp,frame.desp );

    /*
    detector_comp = xfeatures2d::SIFT::create( detector );
    descriptor_comp = xfeatures2d::SIFT::create( descriptor.c_str() );
    if(!detector_comp || !descriptor_comp)
    {
 	cerr<<"Unknown detector or discriptor type !"<<detector<<","<<descriptor<<endl;
	return;
    }
    detector_comp->detect(frame.rgb,frame.kp);
    descriptor_comp->compute(frame.rgb,frame.kp,frame.desp);
    */
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
    int min_good_match=atoi(pd.getData("min_good_match").c_str());
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
    int count=0;
    for(size_t i=0;i<goodMatches.size();i++)
    {
	Point2f p=frame1.kp[goodMatches[i].queryIdx].pt;
	Vec3f pix=frame1.img_3d.at<Vec3f>(p.y,p.x);   
	float x=pix.val[0];
        float y=pix.val[1];
        float z=pix.val[2];
	
	if(1)//frame1.mask.at<float>(p.y,p.x)!=0)//x!=0 || y!=0 || z!=0)
	{
    	    count+=1;
	    Point3f pd (x,y,z);
	    pts_obj.push_back(pd);
	    pts_img.push_back(Point2f(frame2.kp[goodMatches[i].trainIdx].pt));
	}
    }
    cout<<"used goodmatches"<<count<<endl;
    double camera_matrix_data[3][3]={{camera.fx, 0, camera.cx},
				     {0, camera.fy, camera.cy},
				     {0, 0, 1}};
   
    //cout<<"solving pnp"<<endl;
    RESULT_OF_PNP result;
    if(pts_obj.size()<min_good_match)
    {
	Mat tmp=Mat::zeros(Size(3,3),CV_64FC1);
	result.Rvec=tmp;//.at<double>(0,0)=0;
	return result;
    }
    Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    Mat rvec, tvec, inliers;
    //cout<<"before solve"<<endl;
    solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 1.0, 0.99, inliers );
    //cout<<"end solve"<<endl;
    result.Rvec=rvec;
    result.Tvec=tvec;
    result.inliers=inliers.rows;
    //cout<<"R= "<<result.Rvec<<endl;
    //cout<<"T= "<<result.Tvec<<endl;
    //cout<<"Done solving pnnp"<<endl;
    return result;
}

Eigen::Isometry3d cvMat2Eigen(Mat& rvec, Mat& tvec)
{
    Mat R;
    Rodrigues(rvec,R);
    Eigen::Matrix3d r;
    cv2eigen(R,r);
    //cout<<"R="<<r<<endl;
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
    
    return tmp;
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

void get_Camera(ParameterReader pd,Mat cam_mtx1,Mat cam_mtx2,Mat dist1,Mat dist2,Mat R1,Mat P1,Mat R2,Mat P2,Mat Q)
{
    cam_mtx1.at<float>(0,0)=atof((pd.getData("cam1.fx")).c_str());
    cam_mtx1.at<float>(0,2)=atof((pd.getData("cam1.cx")).c_str());
    cam_mtx1.at<float>(1,1)=atof((pd.getData("cam1.fy")).c_str());
    cam_mtx1.at<float>(1,2)=atof((pd.getData("cam1.cy")).c_str());

    cam_mtx2.at<float>(0,0)=atof((pd.getData("cam2.fx")).c_str());
    cam_mtx2.at<float>(0,2)=atof((pd.getData("cam2.cx")).c_str());
    cam_mtx2.at<float>(1,1)=atof((pd.getData("cam2.fy")).c_str());
    cam_mtx2.at<float>(1,2)=atof((pd.getData("cam2.cy")).c_str());

    R1.at<float>(0,0)=atof((pd.getData("R1.00")).c_str());
    R1.at<float>(0,1)=atof((pd.getData("R1.01")).c_str());
    R1.at<float>(0,2)=atof((pd.getData("R1.02")).c_str());
    R1.at<float>(1,0)=atof((pd.getData("R1.10")).c_str());
    R1.at<float>(1,1)=atof((pd.getData("R1.11")).c_str());
    R1.at<float>(1,2)=atof((pd.getData("R1.12")).c_str());
    R1.at<float>(2,0)=atof((pd.getData("R1.20")).c_str());
    R1.at<float>(2,1)=atof((pd.getData("R1.21")).c_str());
    R1.at<float>(2,2)=atof((pd.getData("R1.22")).c_str());

    R2.at<float>(0,0)=atof((pd.getData("R2.00")).c_str());
    R2.at<float>(0,1)=atof((pd.getData("R2.01")).c_str());
    R2.at<float>(0,2)=atof((pd.getData("R2.02")).c_str());
    R2.at<float>(1,0)=atof((pd.getData("R2.10")).c_str());
    R2.at<float>(1,1)=atof((pd.getData("R2.11")).c_str());
    R2.at<float>(1,2)=atof((pd.getData("R2.12")).c_str());
    R2.at<float>(2,0)=atof((pd.getData("R2.20")).c_str());
    R2.at<float>(2,1)=atof((pd.getData("R2.21")).c_str());
    R2.at<float>(2,2)=atof((pd.getData("R2.22")).c_str());

    dist1.at<float>(0,0)=atof((pd.getData("dist1.a")).c_str());
    dist1.at<float>(1,0)=atof((pd.getData("dist1.b")).c_str());
    dist1.at<float>(2,0)=atof((pd.getData("dist1.c")).c_str());
    dist1.at<float>(3,0)=atof((pd.getData("dist1.d")).c_str());
    
    dist2.at<float>(0,0)=atof((pd.getData("dist2.a")).c_str());
    dist2.at<float>(1,0)=atof((pd.getData("dist2.b")).c_str());
    dist2.at<float>(2,0)=atof((pd.getData("dist2.c")).c_str());
    dist2.at<float>(3,0)=atof((pd.getData("dist2.d")).c_str());

    P1.at<float>(0,0)=atof((pd.getData("camera.fx")).c_str());
    P1.at<float>(1,1)=atof((pd.getData("camera.fy")).c_str());
    P1.at<float>(0,2)=atof((pd.getData("camera.cx")).c_str());
    P1.at<float>(1,2)=atof((pd.getData("camera.cy")).c_str());
    P1.at<float>(2,2)=1;

    P2.at<float>(0,0)=atof((pd.getData("camera.fx")).c_str());
    P2.at<float>(1,1)=atof((pd.getData("camera.fy")).c_str());
    P2.at<float>(0,2)=atof((pd.getData("camera.cx")).c_str());
    P2.at<float>(1,2)=atof((pd.getData("camera.cy")).c_str());
    P2.at<float>(2,2)=1;
    P2.at<float>(0,3)=atof((pd.getData("camera.p2")).c_str());

    Q.at<float>(0,0)=1;
    Q.at<float>(1,1)=-1;
    Q.at<float>(0,3)=-atof((pd.getData("camera.cx")).c_str());
    Q.at<float>(1,3)=atof((pd.getData("camera.cy")).c_str());
    Q.at<float>(2,3)=-atof((pd.getData("camera.fx")).c_str());
    Q.at<float>(3,2)=atof((pd.getData("Q32")).c_str());
}
void visualOdometry()
{
    ParameterReader pd;
    PointCloud::Ptr cloud(new PointCloud());
    int startIndex=atoi(pd.getData("start_index").c_str());
    int endIndex=atoi(pd.getData("end_index").c_str());
    double step=atof(pd.getData("step").c_str());
    cout<<"Initializing ..."<<endl;
    int currIndex=startIndex;
	
    string path;
    string detector=pd.getData("detector");
    string descriptor=pd.getData("descriptor");
    stringstream convert;
    convert<<startIndex;
    path=pd.getData("path");
    float pc_thresh=atof(pd.getData("pc_thresh").c_str());
    double count=double(startIndex); 
    double invalid_R[3][3]={{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};
    Mat cam_mtx1,cam_mtx2,dist1,dist2,R1,P1,R2,P2,Q;
    cam_mtx1=Mat::eye(3,3,CV_32FC1);
    cam_mtx2=Mat::eye(3,3,CV_32FC1);
    R1=Mat::eye(3,3,CV_32FC1);
    R2=Mat::eye(3,3,CV_32FC1);
    P1=Mat::zeros(3,4,CV_32FC1);
    P2=Mat::zeros(3,4,CV_32FC1);
    dist1=Mat::zeros(4,1,CV_32FC1);
    dist2=Mat::zeros(4,1,CV_32FC1);
    Q=Mat::zeros(4,4,CV_32FC1);
    get_Camera(pd,cam_mtx1,cam_mtx2,dist1,dist2,R1,P1,R2,P2,Q);
    //Mat tmp_img=imread(path+"left/2.jpg");
    Mat map_x1,map_y1,map_x2,map_y2;
    
    fisheye::initUndistortRectifyMap(cam_mtx1,dist1,R1,P1,Size(752,480),CV_32FC1,map_x1,map_y1);
    fisheye::initUndistortRectifyMap(cam_mtx2,dist2,R2,P2,Size(752,480),CV_32FC1,map_x2,map_y2);
	
    CAMERA_INTRINSIC_PARAMETERS camera=getCamera(pd);
    FRAME lastFrame=image2PointCloud(path,count,pc_thresh,map_x1,map_y1,map_x2,map_y2,Q);
    cloud=lastFrame.cloud;
   
    computeKeyPointsAndDesp( lastFrame, detector, descriptor );
    //cout<<"success compute key."<<endl;
    pcl::visualization::CloudViewer viewer("viewer");
    
    bool visualize = pd.getData("visualize_pointcloud")==string("yes");
    int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    double max_norm = atof( pd.getData("max_norm").c_str() );
   
    if(visualize==true)
	viewer.showCloud(cloud);
    count+=step;
    for(currIndex=startIndex+1;currIndex<endIndex;currIndex++)
    {
	cout<<"Reading files"<<count+1<<endl;
        FRAME currFrame=image2PointCloud(path,count,pc_thresh,map_x1,map_y1,map_x2,map_y2,Q);
	computeKeyPointsAndDesp(currFrame,detector,descriptor);
	RESULT_OF_PNP result=estimateMotion(lastFrame,currFrame,camera);
	cout<<"inliers="<<result.inliers<<endl;
	if(result.Rvec.at<double>(0,0)==0 || result.inliers<min_inliers)
	{
	    count+=step;
	    continue;
	}
	//double norm=normofTransform(result.Rvec,result.Tvec);
	//cout<<"norm="<<norm<<endl;
	cout<<norm(result.Tvec)<<"   "<<norm(result.Rvec)<<endl;
	if(norm(result.Tvec)<=0 || norm(result.Tvec)>1 || norm(result.Rvec)<3.1 || norm(result.Rvec)>3.2)//norm>max_norm || norm<=1)
	{
	    count+=step;
	    continue;
	}
	Eigen::Isometry3d T=cvMat2Eigen(result.Rvec,result.Tvec);
	//cout<<"T="<<T.matrix()<<endl;
	cloud=joinPointCloud(cloud,currFrame,T,camera);
	if(visualize==true)
	    viewer.showCloud(cloud);
	lastFrame=currFrame;
	count+=step;
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
    cout<<norm(tvec)<<"   "<<norm(rvec)<<"   "<<rvec<<endl;
    return (fabs(min(norm(rvec),2*M_PI-norm(rvec)))+fabs(norm(tvec)));
}
