#include<opencv2/opencv.hpp>
#include<iostream>
// #include<opencv2/nonfree/nonfree.hpp>
using namespace cv;
int main()
{
    cv::Mat img1=cv::imread("/home/kuromadoshi/Downloads/desk/DSC02638.JPG",cv::IMREAD_GRAYSCALE);
    cv::Mat img2=cv::imread("/home/kuromadoshi/Downloads/desk/DSC02639.JPG",cv::IMREAD_GRAYSCALE);
    
    resize(img1,img1,img1.size()/10);
    resize(img2,img2,img2.size()/10);
    
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    std::vector<cv::KeyPoint> kp1; //for first image
    std::vector<cv::KeyPoint> kp2; // for second image

    cv::Mat desc1,desc2;
    

    detector->detect(img1,kp1);detector->compute(img1,kp1,desc1);
    detector->detectAndCompute(img2,cv::noArray(),kp2,desc2);
    
    desc1.convertTo(desc1,CV_32F);
    desc2.convertTo(desc2,CV_32F);
    
    std::vector<std::vector<cv::DMatch>> knn_matches;
    /*
    knn_matches stores matches found 
    to it has a size n which is equal to the matches found
    every match ie knn_matches[i] is a vector of 2 elements
    every element ie knn_matches[i][j] (where j is either 0 or 1)
    is of type cv::DMatch

    Note: out of those 2 elements 
    First element stores the best match found
    Second element stores the second best match found

    So a good match is the one where 
    ratio of distance of best match to distance of second best match
    is less than a particular threshhold

    Here distance means Sum of Square differnce between(key point in first image and its supposed matching keypoint in other image)
    this SSD is estimate of error
    so if the value is low it is better 
    */
    matcher->knnMatch(desc1,desc2,knn_matches,2);

    const float ratio_thresh = 0.7f;

    std::vector<DMatch> good_matches;

    
    for(size_t i=0;i< knn_matches.size();i++){
        // std::cout<<knn_matches[i].size();
        if(knn_matches[i][0].distance<ratio_thresh*knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
            using namespace std;
            cout<<"Check1: "<<knn_matches[i][0].queryIdx<<" "<<knn_matches[i][0].trainIdx<<" "<<knn_matches[i][0].imgIdx<<endl;
            cout<<"Check1: "<<knn_matches[i][1].queryIdx<<" "<<knn_matches[i][1].trainIdx<<" "<<knn_matches[i][1].imgIdx<<endl<<endl;
        };
    }
    cv::Mat img_matches;
    cv::drawMatches(img1,kp1,img2,kp2,good_matches,img_matches,cv::Scalar::all(-1),cv::Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Good Matches",img_matches);
    cv::waitKey();
    return 0;
    
}