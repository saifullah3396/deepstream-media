#include <algorithm>
#ifdef DEBUG
#include <chrono>
#endif
#include <cstring>
#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <math.h>
#include <set>
#include <vector>
#include "nvdsinfer_custom_impl.h"

#include <locale>
#include <codecvt>
#include <string>

#define OUTPUT_LAYER_DIMS 2048

static bool configLoaded = false;

static const std::wstring numbers = L"0123456789";
static const std::wstring symbols = L"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
static std::string model_name = "latin";
static std::string lang = "en";
static std::wstring recognizable_chars;
static std::wstring ignore_chars;
static std::vector<std::string> lang_dict;

extern "C" bool loadModelChars(const char *app_root)
{
    Json::Reader reader;
    Json::Value model_configs;

    // load model characters first
    std::ifstream model_configs_file(
        std::string(app_root) +
        "/src/nvdsinfer_parsers/text_recognition/models_config.json");
    if (
        model_configs_file.is_open() &&
        reader.parse(model_configs_file, model_configs))
    {
        bool lang_incompatible = true;
        const auto &model_langs = model_configs[model_name]["langs"];
        for (unsigned int i = 0; i < model_langs.size(); i++)
        {
            if (lang == model_langs[i].asString())
            {
                lang_incompatible = false;
            }
        }

        if (lang_incompatible)
        {
            std::cerr
                << "Input language: "
                << lang
                << " is not compatible with the model: "
                << model_name
                << std::endl;
            return false;
        }

        // load name labels of persons to be recognized
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        const auto &model_chars = converter.from_bytes(model_configs[model_name]["chars"].asString());

        recognizable_chars = numbers + symbols + model_chars;
        return true;
    }
    else
    {
        std::cerr
            << "Text Recognition: Failed to load model characters."
            << std::endl;
        return false;
    }
}

extern "C" bool loadLanguageChars(const char *app_root)
{
    // load language characters
    std::ifstream lang_characters_file(
        std::string(app_root) +
        "/src/nvdsinfer_parsers/text_recognition/characters/" + lang + "_char.txt");
    if (lang_characters_file.is_open())
    {
        // create a set of language characters
        std::wstring lang_chars;
        std::string line;
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        while (std::getline(lang_characters_file, line))
        {
            lang_chars.append(converter.from_bytes(*line.begin()));
        }
        lang_chars += symbols;
        lang_chars += numbers;

        int count = 0;
        ignore_chars.clear();
        for (const auto &rc : recognizable_chars)
        {
            bool remove = true;
            for (const auto &lc : lang_chars)
            {
                if (rc == lc)
                {

                    remove = false;
                }
            }

            if (remove)
            {
                count++;
                ignore_chars += rc;
            }
        }
        return true;
    }
    else
    {
        std::cerr
            << "Text Recognition: Failed to load language characters."
            << std::endl;
        return false;
    }
}

extern "C" bool loadLanguageDict(const char *app_root)
{
    // load language dictionary
    std::ifstream lang_dict_file(
        std::string(app_root) +
        "/src/nvdsinfer_parsers/text_recognition/dict/" + lang + ".txt");
    if (lang_dict_file.is_open())
    {
        // create a set of language characters
        std::string line;
        while (std::getline(lang_dict_file, line))
        {
            lang_dict.push_back(line);
        }
        return true;
    }
    else
    {
        std::cerr
            << "Text Recognition: Failed to load language dictionary."
            << std::endl;
        return false;
    }
}

extern "C" bool NvDsInferTextRecognitionParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferTextRecognitionParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString)
{
#ifdef DEBUG
    auto timeStart = std::chrono::high_resolution_clock::now();
#endif
    if (!configLoaded)
    {
        std::cout
            << "Text Recognition: Loading characters and words dictionary..."
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

        if (!loadModelChars(app_root) ||
            !loadLanguageChars(app_root) ||
            !loadLanguageDict(app_root))
        {
            return false;
        }

        configLoaded = true;
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
            float *outputEmbedding = (float *)outputLayersInfo[lIdx].buffer;
            NvDsInferAttribute attr;

            // find the idx of the letter with maximum probability for each channel
            std::vector<float> maxProb(dims.c, 0.0);
            std::vector<int> maxProbIdx(dims.c, -1);
            for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
            {
                float prob = 0.0;
                for (unsigned int hIdx = 0; hIdx < dims.h; hIdx++)
                {
                    prob = outputEmbedding[cIdx * dims.h + hIdx];
                    if (prob > maxProb[cIdx])
                    {
                        maxProb[cIdx] = prob;
                        maxProbIdx[cIdx] = hIdx;
                    }
                }
            }

            std::wstring output;
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
            {
                const auto &charIdx = maxProbIdx[cIdx];
                if (charIdx > 0 && !(cIdx > 0 && charIdx == maxProbIdx[cIdx - 1]))
                {
                    if (
                        ignore_chars.find(recognizable_chars[charIdx - 1]) ==
                        std::wstring::npos)
                    {
                        output += recognizable_chars[charIdx - 1];
                    }
                }
            }

            if (!output.empty())
            {
                auto label = new std::string(converter.to_bytes(output));
                attr.attributeIndex = 0;
                attr.attributeValue = 0;
                attr.attributeConfidence = 1;
                attr.attributeLabel = label->c_str();
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
            << "NvDsInferTextRecognitionParser: "
            << "Exception raised with the following message: " << exc.what() << std::endl;
        return true;
    }

#ifdef DEBUG
    auto diff =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - timeStart)
            .count();
    std::cout << "Time taken:" << diff << std::endl;
#endif
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferTextRecognitionParser);
