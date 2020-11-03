/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "nvdsinfer_custom_impl.h"

//#define MIN(a,b) ((a) < (b) ? (a) : (b))
//#define MAX(a,b) ((a) > (b) ? (a) : (b))
//#define CLIP(a,min,max) (MAX(MIN(a, max), min))
//#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)
#define TEXT_THRESHOLD 0.7
#define LOW_TEXT_THRESHOLD 0.4
#define LINK_THRESHOLD 0.4
#define SIZE_THRESHOLD 10

/* This is a bounding box parsing function for the CRAFT net
 * detector model for text detection. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCraft (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCraft (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  auto timeStart = std::chrono::high_resolution_clock::now();
  static NvDsInferDimsCHW scoresLayerDims;
  static int scoresLayerIndex = -1;

  /* Find the scores layer */
  if (scoresLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      std::cout << "outputLayersInfo[i].layerName:" << outputLayersInfo[i].layerName << std::endl;
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0) {
        scoresLayerIndex = i;
        getDimsCHWFromDims(scoresLayerDims, outputLayersInfo[i].inferDims);
        break;
      }
    }
    if (scoresLayerIndex == -1) {
    std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
    return false;
    }
  }

  int gridW = scoresLayerDims.c; // channel and width are switched here
  int gridH = scoresLayerDims.h;
  int gridC = scoresLayerDims.w; // channel and width are switched here
  int gridSize = gridW * gridH;
  auto *scoresBuffer = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  auto textMap = cv::Mat(gridW, gridH, CV_32FC1, scoresBuffer);
  auto linkMap = cv::Mat(gridW, gridH, CV_32FC1, scoresBuffer + gridW * gridH);

  cv::Mat textScore, linkScore;
  cv::threshold(textMap, textScore, LOW_TEXT_THRESHOLD, 1, 0);
  cv::threshold(linkMap, linkScore, LINK_THRESHOLD, 1, 0);
  cv::Mat textScoreComb = cv::min(cv::max(textMap + linkMap, 0), 1);
  textScoreComb.convertTo(textScoreComb, CV_8UC1);
  cv::Mat labels, stats, centroids;
  int nLabels = cv::connectedComponentsWithStats(
    textScoreComb, labels, stats, centroids, 4);
  std::vector<cv::Rect> bboxes;
  std::cout << "nLabels:" << nLabels << std::endl;
  for (int i = 0; i < nLabels; ++i) {
    auto size = stats.at<unsigned>(i, cv::CC_STAT_AREA);
    if (size < SIZE_THRESHOLD) continue;
    cv::Mat segMap = (labels==i);
    cv::Mat textMapMasked;
    textMap.copyTo(textMapMasked, segMap);
    double minVal, maxVal;\
    cv::Point minLoc, maxLoc; 
    cv::minMaxLoc(textMapMasked, &minVal, &maxVal, &minLoc, &maxLoc);
    std::cout << "maxVal:" << maxVal << std::endl;
    if (maxVal < TEXT_THRESHOLD) continue;

    cv::Mat segMapMask;
    bitwise_and(linkScore==1, textScore==0, segMapMask);
    segMapMask = 255 - segMapMask; // invert it
    cv::Mat segMapMasked;
    segMap.copyTo(segMapMasked, segMapMask);
    auto x = stats.at<unsigned>(i, cv::CC_STAT_LEFT);
    auto y = stats.at<unsigned>(i, cv::CC_STAT_TOP);
    auto w = stats.at<unsigned>(i, cv::CC_STAT_WIDTH);
    auto h = stats.at<unsigned>(i, cv::CC_STAT_HEIGHT);
    int niter = int(cv::sqrt(size * cv::min(w, h) / (w * h)) * 2);
    int sx = x - niter;
    int ex = x + w + niter + 1;
    int sy = y - niter;
    int ey = y + h + niter + 1;

    cv::min(sx, 0);
    cv::min(sy, 0);
    cv::max(ex, gridW);
    cv::max(ey, gridH);
    auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1+niter, 1+niter));

    auto roiRect = cv::Rect(cv::Point(sx, sy), cv::Point(ex, ey));
    cv::Mat roi(segMapMasked(roiRect));
    cv::dilate(roi, roi, kernel);

    cv::Mat nonZeroLocs;
    cv::findNonZero(roi, nonZeroLocs);
    auto rectangle = cv::minAreaRect(nonZeroLocs);
    cv::Mat box;
    cv::boxPoints(rectangle, box);
    std::cout << "box:" << box << std::endl;
    // auto w = cv::norm(box[0] - box[1]);
    // auto h = cv::norm(box[1] - box[2]);
    // auto boxRatio = cv::max(w, h) / (cv::min(w, h) + 1e-5);
  //   if abs(1 - boxRatio) <= 0.1:
  //       l, r = min(np_contours[:,0]), max(np_contours[:,0])
  //       t, b = min(np_contours[:,1]), max(np_contours[:,1])
  //       box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

  //   # make clock-wise order
  //   startidx = box.sum(axis=1).argmin()
  //   box = np.roll(box, 4-startidx, 0)
  //   box = np.array(box)

  //   det.append(box)
  //   mapper.append(k)
  }
  auto diff = 
    std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - timeStart).count();
  if (bboxes.size() > 0)
    std::cout << "Time taken:" << diff << std::endl;
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCraft);
