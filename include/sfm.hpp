#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>

#define dbgLine std::cerr<<"LINE:"<<__LINE__<<"\n"
#define dbg(x) std::cerr<<(#x)<<" is "<< x <<"\n"


const int DS_FACTOR = 10;
const double FOCAL_LENGTH = 4308/DS_FACTOR ; 
const int MIN_LANDMARK_SEEN = 3;

const std::string IMAGE_PATH="/home/kuromadoshi/Downloads/desk/";

const std::vector<std::string> IMAGES = {
    "DSC02638.JPG",
    "DSC02639.JPG",
    "DSC02640.JPG",
    "DSC02641.JPG",
    "DSC02642.JPG"
};

class ImgPose
{
    public:
        cv::Mat img, desc;
        std::vector<cv::KeyPoint> kp;
        
        cv::Mat T;
        cv::Mat P;

        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // keypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; };
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; };

};

struct Landmark
{
    cv::Point3f pt;
    int seen = 0;
};
std::vector<ImgPose> imgPoses;
std::vector<Landmark> landmark;
