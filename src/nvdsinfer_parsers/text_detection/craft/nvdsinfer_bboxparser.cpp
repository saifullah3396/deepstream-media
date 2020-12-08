#ifdef DEBUG
#include <chrono>
#endif
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "nvdsinfer_custom_impl.h"

#define TEXT_THRESHOLD 0.7
#define LOW_TEXT_THRESHOLD 0.4
#define LINK_THRESHOLD 0.4
#define SIZE_THRESHOLD 10
#define BBOX_RATIO_THRESHOLD 0.1
#define HEIGHT_THRESHOLD 0.5
#define WIDTH_THRESHOLD 1.0
#define Y_CENTERS_THRESHOLD 0.5
#define ADD_MARGIN 0.05

/* This is a bounding box parsing function for the CRAFT net
 * detector model for text detection. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCraft(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C" bool NvDsInferParseCraft(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
#ifdef DEBUG
    auto timeStart = std::chrono::high_resolution_clock::now();
#endif
    static NvDsInferDimsCHW scoresLayerDims;
    static int scoresLayerIndex = -1;

    /* Find the scores layer */
    if (scoresLayerIndex == -1)
    {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++)
        {
            if (strcmp(outputLayersInfo[i].layerName, "scores") == 0)
            {
                scoresLayerIndex = i;
                getDimsCHWFromDims(
                    scoresLayerDims, outputLayersInfo[i].inferDims);
                break;
            }
        }
        if (scoresLayerIndex == -1)
        {
            std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
            return false;
        }
    }

    try
    {
        int gridW = scoresLayerDims.w;
        int gridH = scoresLayerDims.h;
        float ratioW = networkInfo.width / ((float)gridW);
        float ratioH = networkInfo.height / ((float)gridH);
        float *scoresBuffer = (float *)outputLayersInfo[scoresLayerIndex].buffer;
        auto textMap = cv::Mat(cv::Size(gridW, gridH), CV_32FC1, scoresBuffer);
        auto linkMap = cv::Mat(cv::Size(gridW, gridH), CV_32FC1, scoresBuffer + gridW * gridH);

        cv::Mat textScore, linkScore;
        cv::threshold(textMap, textScore, LOW_TEXT_THRESHOLD, 1, 0);
        cv::threshold(linkMap, linkScore, LINK_THRESHOLD, 1, 0);
        cv::Mat textScoreComb = cv::min(cv::max(textMap + linkMap, 0), 1);
        textScoreComb.convertTo(textScoreComb, CV_8UC1);
        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(
            textScoreComb, labels, stats, centroids, 4);
        std::vector<cv::Rect> bboxes;
        for (int i = 0; i < nLabels; ++i)
        {
            auto size = stats.at<unsigned>(i, cv::CC_STAT_AREA);
            if (size < SIZE_THRESHOLD)
                continue;
            cv::Mat segMap = (labels == i);
            cv::Mat textMapMasked;
            textMap.copyTo(textMapMasked, segMap);
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(textMapMasked, &minVal, &maxVal, &minLoc, &maxLoc);
            if (maxVal < TEXT_THRESHOLD)
                continue;

            cv::Mat segMapMask;
            bitwise_and(linkScore == 1, textScore == 0, segMapMask);
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
            auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1 + niter, 1 + niter));

            auto roiRect = cv::Rect(cv::Point(sx, sy), cv::Point(ex, ey));
            cv::Mat roi(segMapMasked(roiRect));
            cv::dilate(roi, roi, kernel);

            std::vector<cv::Point> nonZeroLocsVec;
            cv::findNonZero(roi, nonZeroLocsVec);
            auto rectangle = cv::minAreaRect(nonZeroLocsVec);
            cv::Mat box;
            cv::boxPoints(rectangle, box);
            auto boxW = cv::norm(box.at<float>(0, 0) - box.at<float>(3, 0));
            auto boxH = cv::norm(box.at<float>(0, 1) - box.at<float>(1, 1));
            auto boxRatio = cv::max(boxW, boxH) / (cv::min(boxW, boxH) + 1e-5);
            cv::Rect boxRect;
            if (fabsf(1.0 - boxRatio) <= BBOX_RATIO_THRESHOLD)
            {
                cv::Mat nonZeroLocs =
                    cv::Mat(
                        nonZeroLocsVec.size(), 2, CV_32SC1, nonZeroLocsVec.data());
                nonZeroLocs.convertTo(nonZeroLocs, CV_8U);
                cv::Mat minXY, maxXY;
                cv::reduce(nonZeroLocs, minXY, 0, CV_REDUCE_MIN);
                cv::reduce(nonZeroLocs, maxXY, 0, CV_REDUCE_MAX);

                boxRect = cv::Rect(cv::Point(minXY), cv::Point(maxXY));

                // translate the box to start of roi
                boxRect.x += sx;
                boxRect.y += sy;

                // scale the box to original image size
                boxRect.x *= ratioW;
                boxRect.y *= ratioH;
                boxRect.width *= ratioW;
                boxRect.height *= ratioH;
            }
            else
            {
                boxRect = cv::boundingRect(box);

                // translate the box to start of roi
                boxRect.x += sx;
                boxRect.y += sy;

                // scale the box to original image size
                boxRect.x *= ratioW;
                boxRect.y *= ratioH;
                boxRect.width *= ratioW;
                boxRect.height *= ratioH;
            }
            bboxes.push_back(boxRect);
        }

        // create horizontal lists as done in easyocr but only considering
        // boxes, not polys
        std::vector<cv::Rect> &horizontalList = bboxes;
        std::sort(
            horizontalList.begin(), horizontalList.end(),
            [](const cv::Rect &r1, const cv::Rect &r2) -> bool {
                // sort based on y center
                return r1.y + 0.5 * r1.height < r2.y + 0.5 * r2.height;
            });

        std::vector<cv::Rect> mergableBoxes;
        std::vector<std::vector<cv::Rect>> mergableBoxesList;
        const auto &firstBBox = horizontalList[0];
        float mergableBoxYCenters = firstBBox.y + 0.5 * firstBBox.height;
        float mergableBoxHeights = firstBBox.height;
        mergableBoxes.push_back(firstBBox);
        int meanN = 1;
        for (size_t i = 1; i < horizontalList.size(); ++i) // starts from 1
        {
            const auto &bbox = horizontalList[i];
            const auto bboxYCenter = bbox.y + 0.5 * bbox.height;
            mergableBoxYCenters += bboxYCenter;
            mergableBoxHeights += bbox.height;
            meanN++;

            auto yCentersMean = mergableBoxYCenters / ((float)meanN);
            auto heightsMean = mergableBoxHeights / ((float)meanN);
            auto cond1 =
                fabsf(heightsMean - bbox.height) <
                HEIGHT_THRESHOLD * heightsMean;
            auto cond2 =
                fabsf(yCentersMean - bboxYCenter) <
                Y_CENTERS_THRESHOLD * heightsMean;
            if (cond1 && cond2)
            {
                mergableBoxes.push_back(bbox);
            }
            else
            {
                mergableBoxesList.push_back(mergableBoxes);

                mergableBoxYCenters = bboxYCenter;
                mergableBoxHeights = bbox.height;
                meanN = 1;

                mergableBoxes.clear();
                mergableBoxes.push_back(bbox);
            }
        }
        mergableBoxesList.push_back(mergableBoxes);

        for (auto &bboxList : mergableBoxesList)
        {
            if (bboxList.size() == 1)
            {
                const auto &box = bboxList[0];
                auto margin = ADD_MARGIN * box.height;
                NvDsInferObjectDetectionInfo object;
                object.classId = 0;
                object.detectionConfidence = 1.0;

                object.left = box.x - margin;
                object.top = box.y - margin;
                object.width = box.width + margin;
                object.height = box.height + margin;
                objectList.push_back(object);
            }
            else
            {
                std::sort(
                    bboxList.begin(), bboxList.end(),
                    [](const cv::Rect &r1, const cv::Rect &r2) -> bool {
                        // sort based on y center
                        return r1.x < r2.x;
                    });

                std::vector<cv::Rect> mergedBoxes;
                cv::Rect mergedBox = bboxList[0];
                float x_max = bboxList[0].x + bboxList[0].width;
                for (size_t i = 1; i < bboxList.size(); ++i) // starts from 1
                {
                    const auto &box = bboxList[i];

                    // mistake in easy ocr i think. they are multiplying width
                    // threshold with height instead of width
                    if (fabsf(box.x - x_max) < WIDTH_THRESHOLD * box.width)
                    {
                        mergedBox = mergedBox | box; // take union of the rects
                    }
                    else
                    {
                        mergedBoxes.push_back(mergedBox);
                        mergedBox = box;
                    }
                    x_max = box.x + box.width;
                }
                mergedBoxes.push_back(mergedBox);

                for (const auto &box : mergedBoxes)
                {
                    auto margin = ADD_MARGIN * box.height;
                    NvDsInferObjectDetectionInfo object;
                    object.classId = 0;
                    object.detectionConfidence = 1.0;

                    object.left = box.x - margin;
                    object.top = box.y - margin;
                    object.width = box.width + margin;
                    object.height = box.height + margin;
                    objectList.push_back(object);
                }
            }
        }
    }
    catch (const cv::Exception &exception)
    {
        return true;
    }
#ifdef DEBUG
    auto diff =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count();
    if (objectList.size() > 0)
        std::cout << "Time taken:" << diff << std::endl;
#endif
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCraft);
