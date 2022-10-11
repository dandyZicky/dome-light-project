#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>

#define CANDI_SRC "../images/september/candi/"
#define KOIN_SRC "../images/september/koin/"
#define BUDDHA_SRC "../images/september/buddha/"

cv::VideoCapture captureDevice;

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

void exportMesh(cv::Mat Depth, cv::Mat Normals, cv::Mat texture, const char* result)
{
  /* writing obj for export */
  std::ofstream objFile, mtlFile;
  objFile.open(std::string(result) + "export.obj");
  int width = Depth.cols;
  int height = Depth.rows;
  /* vertices, normals, texture coords */
  objFile << "mtllib export.mtl" << std::endl;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      objFile << "v " << x << " " << y << " "
              << Depth.at<float>(cv::Point(x, y)) << std::endl;
      objFile << "vt " << x / (width - 1.0f) << " " << (1.0f - y) / height
              << " "
              << "0.0" << std::endl;
      objFile << "vn " << (float)Normals.at<cv::Vec3b>(y, x)[0] << " "
              << (float)Normals.at<cv::Vec3b>(y, x)[1] << " "
              << (float)Normals.at<cv::Vec3b>(y, x)[2] << std::endl;
    }
  }

  /* faces */
  objFile << "usemtl picture" << std::endl;

  for (int y = 0; y < height - 1; y++)
  {
    for (int x = 0; x < width - 1; x++)
    {
      int f1 = x + y * width + 1;
      int f2 = x + y * width + 2;
      int f3 = x + (y + 1) * width + 1;
      int f4 = x + (y + 1) * width + 2;
      objFile << "f " << f1 << "/" << f1 << "/" << f1 << " ";
      objFile << f2 << "/" << f2 << "/" << f2 << " ";
      objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
      objFile << "f " << f2 << "/" << f2 << "/" << f2 << " ";
      objFile << f4 << "/" << f4 << "/" << f4 << " ";
      objFile << f3 << "/" << f3 << "/" << f3 << std::endl;
    }
  }
  /* texture */
  cv::imwrite(std::string(result) + "export.jpg", texture);
  mtlFile.open(std::string(result) + "export.mtl");
  mtlFile << "newmtl picture" << std::endl;
  mtlFile << "map_Kd export.jpg" << std::endl;
  objFile.close();
  mtlFile.close();
}

cv::Mat imageMask(std::vector<cv::Mat> camImages)
{
  assert(camImages.size() > 0);
  cv::Mat image1 = camImages[0].clone();
  cv::Mat result;
  int width = image1.size().width;
  int height = image1.size().height;
  printf("width: %d, height: %d\n", width, height);
  cv::Mat blank(height, width, CV_8UC1, cv::Scalar(0));
  // cv::Point p1(width*1/4, height*1/4);
  // cv::Point p2(width*3/4, height*3/4);
  cv::Point p1(0, 0);
  cv::Point p2(width, height);
  cv::Rect rect(p1 ,p2);
  cv::rectangle(blank, rect, cv::Scalar(255), -1);
  return blank;
}

cv::Mat computeNormals(std::vector<cv::Mat> camImages,
                       cv::Mat Mask = cv::Mat(), const char* result = "")
{
  int height = camImages[0].rows;
  int width = camImages[0].cols;
  int numImgs = camImages.size();
  /* populate A */
  cv::Mat A(height * width, numImgs, CV_32FC1, cv::Scalar::all(0));

  for (int k = 0; k < numImgs; k++)
  {
    int idx = 0;

    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        A.at<float>(idx++, k) = camImages[k].data[i * width + j] *
                                sgn(Mask.at<uchar>(cv::Point(j, i)));
      }
    }
  }

  /* speeding up computation, SVD from A^TA instead of AA^T */
  cv::Mat U, S, Vt;
  cv::SVD::compute(A.t(), S, U, Vt, cv::SVD::MODIFY_A);
  cv::Mat EV = Vt.t();
  cv::Mat N(height, width, CV_8UC3, cv::Scalar::all(0));
  int idx = 0;

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      if (Mask.at<uchar>(cv::Point(j, i)) == 0)
      {
        N.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
      }
      else
      {
        float rSxyz = 1.0f / sqrt(EV.at<float>(idx, 0) * EV.at<float>(idx, 0) +
                                  EV.at<float>(idx, 1) * EV.at<float>(idx, 1) +
                                  EV.at<float>(idx, 2) * EV.at<float>(idx, 2));
        /* V contains the eigenvectors of A^TA, which are as well the z,x,y
         * components of the surface normals for each pixel	*/
        float sz = 128.0f +
                   127.0f * sgn(EV.at<float>(idx, 0)) *
                       fabs(EV.at<float>(idx, 0)) * rSxyz;
        float sx = 128.0f +
                   127.0f * sgn(EV.at<float>(idx, 1)) *
                       fabs(EV.at<float>(idx, 1)) * rSxyz;
        float sy = 128.0f +
                   127.0f * sgn(EV.at<float>(idx, 2)) *
                       fabs(EV.at<float>(idx, 2)) * rSxyz;
        N.at<cv::Vec3b>(i, j) = cv::Vec3b(sx, sy, sz);
      }

      idx += 1;
    }
  }
  cv::Mat exportNormal;
  cv::cvtColor(N, exportNormal, cv::COLOR_BGR2RGB);
  cv::imwrite(std::string(result) + "export_normal.jpg", exportNormal);
  return N;
}

void updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations)
{
  for (int k = 0; k < iterations; k++)
  {
    for (int i = 1; i < Normals.rows - 1; i++)
    {
      for (int j = 1; j < Normals.cols - 1; j++)
      {
        float zU = Z.at<float>(cv::Point(j, i - 1));
        float zD = Z.at<float>(cv::Point(j, i + 1));
        float zL = Z.at<float>(cv::Point(j - 1, i));
        float zR = Z.at<float>(cv::Point(j + 1, i));
        float nxC = Normals.at<cv::Vec3b>(cv::Point(j, i))[0];
        float nyC = Normals.at<cv::Vec3b>(cv::Point(j, i))[1];
        float nxU = Normals.at<cv::Vec3b>(cv::Point(j, i - 1))[0];
        float nyU = Normals.at<cv::Vec3b>(cv::Point(j, i - 1))[1];
        float nxD = Normals.at<cv::Vec3b>(cv::Point(j, i + 1))[0];
        float nyD = Normals.at<cv::Vec3b>(cv::Point(j, i + 1))[1];
        float nxL = Normals.at<cv::Vec3b>(cv::Point(j - 1, i))[0];
        float nyL = Normals.at<cv::Vec3b>(cv::Point(j - 1, i))[1];
        float nxR = Normals.at<cv::Vec3b>(cv::Point(j + 1, i))[0];
        float nyR = Normals.at<cv::Vec3b>(cv::Point(j + 1, i))[1];
        int up = nxU == 0 && nyU == 0 ? 0 : 1;
        int down = nxD == 0 && nyD == 0 ? 0 : 1;
        int left = nxL == 0 && nyL == 0 ? 0 : 1;
        int right = nxR == 0 && nyR == 0 ? 0 : 1;

        if (up > 0 && down > 0 && left > 0 && right > 0)
        {
          Z.at<float>(cv::Point(j, i)) =
              1.0f / 4.0f * (zD + zU + zR + zL + nxU - nxC + nyL - nyC);
        }
      }
    }
  }
}

cv::Mat cvtFloatToGrayscale(cv::Mat F, int limit = 255)
{
  double min, max;
  cv::minMaxIdx(F, &min, &max);
  cv::Mat adjMap;
  cv::convertScaleAbs(F, adjMap, limit / max);
  return adjMap;
}

cv::Mat localHeightfield(cv::Mat Normals)
{
  const int pyramidLevels = 4;
  const int iterations = 50000;
  /* building image pyramid */
  std::vector<cv::Mat> pyrNormals;
  cv::Mat Normalmap = Normals.clone();
  pyrNormals.push_back(Normalmap);

  for (int i = 0; i < pyramidLevels; i++)
  {
    cv::pyrDown(Normalmap, Normalmap);
    pyrNormals.push_back(Normalmap.clone());
  }

  /* updating depth map along pyramid levels, starting with smallest level at
   * top */
  cv::Mat Z(pyrNormals[pyramidLevels - 1].rows,
            pyrNormals[pyramidLevels - 1].cols, CV_32FC1, cv::Scalar::all(0));

  for (int i = pyramidLevels - 1; i > 0; i--)
  {
    updateHeights(pyrNormals[i], Z, iterations);
    printf("Pyramid-%d\n", i);
    cv::pyrUp(Z, Z);
  }
  printf("Done updating height\n");

  /* linear transformation of matrix values from [min,max] -> [a,b] */
  double min, max;
  cv::minMaxIdx(Z, &min, &max);
  double a = 0.0, b = 50.0;

  for (int i = 0; i < Normals.rows; i++)
  {
    for (int j = 0; j < Normals.cols; j++)
    {
      Z.at<float>(cv::Point(j, i)) =
          (float)a + (b - a) * ((Z.at<float>(cv::Point(j, i)) - min) / (max - min));
    }
  }
  printf("Done computing height\n");

  return Z;
}

int main(int argc, char *argv[])
{
  auto result = KOIN_SRC;
  int numPics = 4;
  std::vector<cv::Mat> camImages;
 
  /* using asset images */
  for (int i = 1; i <= numPics; i++)
  {
    if(i > 9) break;
    std::stringstream s;
    s << result << "Image_0" << i << ".jpg";
    auto imagePush = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
    double scaleDownFactor = 4;
    auto width = imagePush.size().width/scaleDownFactor;
    auto height = imagePush.size().height/scaleDownFactor;
    cv::Mat resized_down;
    cv::resize(imagePush, resized_down, cv::Size(width, height));
    camImages.push_back(resized_down);
  }
  if (numPics > 9) {
    for (int i = 10; i < numPics+11; i++) {
      std::stringstream s;
      s << result << "Image_" << i << ".jpg";
      auto imagePush = cv::imread(s.str(), cv::IMREAD_GRAYSCALE);
      double scaleDownFactor = 4;
      auto width = imagePush.size().width/scaleDownFactor;
      auto height = imagePush.size().height/scaleDownFactor;
      cv::Mat resized_down;
      cv::resize(imagePush, resized_down, cv::Size(width, height));
      camImages.push_back(resized_down);
    }
  }

  /* threshold images */
  cv::Mat Mask = imageMask(camImages);
  for (int i = numPics; i > 0; i--)
  {
    cv::String namaWindow = "img" + std::to_string(i);
    cv::namedWindow(namaWindow, cv::WINDOW_NORMAL);
    printf("Chilling\n");
    cv::imshow(namaWindow, camImages[numPics-i]);
  }

  cv::imshow("Mask", Mask);

  /* compute normal map */
  cv::Mat S = computeNormals(camImages, Mask , result);
  cv::Mat Normalmap;
  cv::cvtColor(S, Normalmap, cv::COLOR_BGR2RGB);
  cv::imshow("normalmap.jpeg", Normalmap);

  /* compute depth map */
  cv::Mat Depth = localHeightfield(S);
  exportMesh(Depth, S, camImages[0], result);
  cv::imshow("Local Depthmap", cvtFloatToGrayscale(Depth));
  cv::imwrite(cv::String(std::string(result) + "export_depth.jpg"), cvtFloatToGrayscale(Depth));
  cv::waitKey(0);
  return 0;
}
