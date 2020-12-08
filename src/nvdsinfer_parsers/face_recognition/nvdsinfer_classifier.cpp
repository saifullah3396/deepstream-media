#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <math.h>
#include <jsoncpp/json/json.h>
#include "nvdsinfer_custom_impl.h"

#define OUTPUT_LAYER_DIMS 2048

static bool embeddingsLoaded = false;
static std::vector<std::string> labels;
static std::vector<std::vector<float>> knownEmbeddings;

extern "C" bool NvDsInferFaceRecognitionParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferFaceRecognitionParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
    if (!embeddingsLoaded)
    {
        std::cout
            << "Face Recognition: Loading embeddings dictionary..."
            << std::endl;
        const char *app_root = std::getenv("MEDIA_APP_ROOT");
        if (!app_root)
        {
            std::cerr
                << "Please add the environment variable MEDIA_APP_ROOT "
                << "pointing to the root of media_app_deployment directory."
                << std::endl;
            return false;
        }

        std::ifstream dictionary_path(
            std::string(app_root) +
            "/src/nvdsinfer_parsers/face_recognition/embeddings.json");
        Json::Reader reader;
        Json::Value dict;
        if (dictionary_path.is_open() && reader.parse(dictionary_path, dict))
        {
            // load name labels of persons to be recognized
            const auto &names = dict["names"];
            for (unsigned int i = 0; i < names.size(); i++)
            {
                labels.push_back(names[i].asString());
            }

            // load embeddings of persons to be recognized
            const auto &embeddings = dict["embeddings"];
            for (unsigned int i = 0; i < embeddings.size(); i++)
            {
                knownEmbeddings.push_back(std::vector<float>());
                for (unsigned int j = 0; j < embeddings[i].size(); j++)
                {
                    knownEmbeddings[i].push_back(embeddings[i][j].asFloat());
                }
            }
        }
        else
        {
            std::cerr
                << "Face Recognition: Failed to load embeddings dictionary."
                << std::endl;
            return false;
        }
        embeddingsLoaded = true;
    }

    try
    {
        /* Get the number of attributes supported by the classifier. */
        unsigned int numAttributes = outputLayersInfo.size();

        /* Iterate through all the output coverage layers of the classifier.
        */
        for (unsigned int lIdx = 0; lIdx < numAttributes; lIdx++)
        {
            /* outputCoverageBuffer for classifiers is usually a softmax layer.
            * The layer is an array of probabilities of the object belonging
            * to each class with each probability being in the range [0,1] and
            * sum all probabilities will be 1.
            */
            NvDsInferDimsCHW dims;
            getDimsCHWFromDims(dims, outputLayersInfo[lIdx].inferDims);
            unsigned int outputEmbeddingSize = dims.c;
            float *outputEmbedding = (float *)outputLayersInfo[lIdx].buffer;
            float &correlationThreshold = classifierThreshold;
            NvDsInferAttribute attr;

            int bestDist = 1000;
            int bestIdx = -1;
            for (unsigned int kIdx = 0; kIdx < knownEmbeddings.size(); ++kIdx)
            {
                float prodMean = 0.0;
                float outputSqrdMean = 0.0;
                float knownSqrdMean = 0.0;
                for (unsigned int jIdx = 0; jIdx < outputEmbeddingSize; jIdx++)
                {
                    auto ke = knownEmbeddings[kIdx][jIdx];
                    auto oe = outputEmbedding[jIdx];
                    prodMean += (ke * oe);
                    outputSqrdMean += (ke * ke);
                    knownSqrdMean += (oe * oe);
                }
                prodMean /= outputEmbeddingSize;
                outputSqrdMean /= outputEmbeddingSize;
                knownSqrdMean /= outputEmbeddingSize;
                auto dist = 1.0 - prodMean / sqrt(outputSqrdMean * knownSqrdMean);
                if (dist == dist && dist < correlationThreshold && dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = kIdx;
                }
            }

            if (bestIdx != -1)
            {
                attr.attributeIndex = lIdx;
                attr.attributeValue = bestIdx;
                attr.attributeConfidence = bestDist;
                attr.attributeLabel = labels[bestIdx].c_str();
                attrList.push_back(attr);
                if (attr.attributeLabel)
                {
                    descString = attr.attributeLabel;
                    descString.append(" ");
                }
            }
        }
    }
    catch (const std::exception &exc)
    {
        std::cerr
            << "NvDsInferFaceRecognitionParser: "
            << "Exception raised with the following message: " << exc.what() << std::endl;
        return true;
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferFaceRecognitionParser);
