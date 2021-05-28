#include <algorithm>
#include <boost/algorithm/string.hpp>
#ifdef DEBUG
#include <chrono>
#endif
#include <codecvt>
#include <iostream>
#include <jsoncpp/json/json.h>
#include "nvdsinfer_custom_impl.h"
#include "ctc_decoder/ctc_beam_search_decoder.h"
#include "ctc_decoder/ctc_greedy_decoder.h"
#include "text_post_processor.h"

#define OUTPUT_LAYER_DIMS 2048

static bool configLoaded = false;
static const std::string numbers = "0123456789";
static const std::string symbols = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
static std::vector<std::string> model_recognizable_chars;
static std::vector<unsigned int> ignore_idx;
static std::string parser_lang;
static bool rtl = false;

/**
 * Loads all the characters that are recognizable by the model of given name
 * @param app_root: Path to the root of application for relative path sub
 * @param model_name: Name of the model
 * @param lang: Language name to load with this modle
 */
extern "C" bool loadModelChars(
    const char *app_root,
    const std::string &model_name = "latin",
    const std::string &lang = "en")
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
    else
    {
      parser_lang = lang;
    }

    // load left to right config
    rtl = model_configs[model_name]["rtl"].asBool();

    // load language characters
    auto lang_chars =
        model_configs[model_name]["lang_chars"][lang].asString();
    lang_chars = numbers + symbols + lang_chars;

    std::vector<std::string> lang_chars_vec;
    for (const auto &ch : lang_chars)
    {
      lang_chars_vec.push_back(std::string(1, ch));
    }

    // load model characters
    auto model_chars = model_configs[model_name]["chars"].asString();
    model_chars = numbers + symbols + model_chars;

    model_recognizable_chars.clear();
    for (const auto &ch : model_chars)
    {
      model_recognizable_chars.push_back(std::string(1, ch));
    }

    for (const auto &ch : lang_chars)
    {
      lang_chars_vec.push_back(std::string(1, ch));
    }

    ignore_idx.clear();
    for (unsigned int i = 0; i < model_recognizable_chars.size(); ++i)
    {
      bool remove = true;
      const auto &rc = model_recognizable_chars[i];
      for (const auto &lc : lang_chars_vec)
      {
        if (rc == lc)
        {
          remove = false;
        }
      }

      if (remove)
      {
        ignore_idx.push_back(i);
      }
    }

    std::vector<std::string> model_recognizable_chars_updated;
    for (unsigned int i = 0; i < model_recognizable_chars.size(); ++i)
    {
      if (std::find(ignore_idx.begin(), ignore_idx.end(), i) == ignore_idx.end())
      {
        model_recognizable_chars_updated.push_back(model_recognizable_chars[i]);
      }
    }
    model_recognizable_chars = model_recognizable_chars_updated;
    // std::cout << "model_characters_updated:" << std::endl;
    // for (unsigned int i = 0; i < model_recognizable_chars.size(); ++i)
    // {
    //     std::cout << model_recognizable_chars[i] << ",";
    // }
    // std::cout << std::endl;
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

extern "C" bool NvDsInferTextRecognitionArabicParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferTextRecognitionArabicParser(
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
        << "NvDsInferTextRecognitionArabicParser: Loading characters and words dictionary..."
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

    if (!loadModelChars(app_root, "arabic", "ur") ||
        !TextPostProcessor::loadSpellCheckerDictionary(app_root, "ur"))
    {
      return false;
    }
    std::cout
        << "NvDsInferTextRecognitionArabicParser: configuration loaded."
        << std::endl;
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

#ifdef DEBUG
      const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
#endif
      // take only the data which is not to be ignored
      std::vector<std::vector<double> > mat_data;
      for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
      {
        mat_data.push_back(std::vector<double>());
        for (unsigned int hIdx = 1; hIdx < dims.h; hIdx++)
        {
          if (std::find(ignore_idx.begin(), ignore_idx.end(), hIdx - 1) == ignore_idx.end())
          {
            mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h + hIdx]);
          }
        }
        mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h]);
      }
      std::string results =
          ctc_greedy_decoder(mat_data, model_recognizable_chars);
      std::string output = results;
      // std::vector<std::pair<double, std::string>> results =
      //     ctc_beam_search_decoder(
      //         mat_data,
      //         model_recognizable_chars,
      //         5);
      // std::string output = results[0].second;
#ifdef DEBUG
      const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
      std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime - startTime).count() << "ms\n\n";
#endif

      if (!output.empty())
      {
        auto label = new std::string(output);
        // *label = TextPostProcessor::processText(*label, parser_lang);
        std::cout << "label:" << label->c_str();
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
        << "NvDsInferTextRecognitionLatinParser: "
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
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferTextRecognitionArabicParser);

extern "C" bool NvDsInferTextRecognitionLatinParser(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &descString);

extern "C" bool NvDsInferTextRecognitionLatinParser(
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
        << "NvDsInferTextRecognitionLatinParser: Loading characters and words dictionary..."
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
        !TextPostProcessor::loadSpellCheckerDictionary(app_root))
    {
      return false;
    }
    std::cout
        << "NvDsInferTextRecognitionLatinParser: configuration loaded."
        << std::endl;
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

#ifdef DEBUG
      const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
#endif
      // take only the data which is not to be ignored
      std::vector<std::vector<double> > mat_data;
      for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
      {
        mat_data.push_back(std::vector<double>());
        for (unsigned int hIdx = 1; hIdx < dims.h; hIdx++)
        {
          if (std::find(ignore_idx.begin(), ignore_idx.end(), hIdx - 1) == ignore_idx.end())
          {
            mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h + hIdx]);
          }
        }
        mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h]);
      }
      std::string results =
          ctc_greedy_decoder(mat_data, model_recognizable_chars);
      std::string output = results;
      // std::vector<std::pair<double, std::string>> results =
      //     ctc_beam_search_decoder(
      //         mat_data,
      //         model_recognizable_chars,
      //         5);
      // std::string output = results[0].second;
#ifdef DEBUG
      const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
      std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime - startTime).count() << "ms\n\n";
#endif

      if (!output.empty())
      {
        auto label = new std::string(output);
        //*label = TextPostProcessor::processText(*label, parser_lang);
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
        << "NvDsInferTextRecognitionLatinParser: "
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
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferTextRecognitionLatinParser);