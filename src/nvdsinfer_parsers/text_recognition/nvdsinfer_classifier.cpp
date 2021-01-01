#include "ctc_decode/ctc_beam_search_decoder.h"
#include "ctc_decode/ctc_greedy_decoder.h"
#include <algorithm>
#include <boost/algorithm/string.hpp>
#ifdef DEBUG
#include <chrono>
#endif
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <math.h>
#include <set>
#include <vector>
#include "nvdsinfer_custom_impl.h"

#include <clocale>
#include <locale>
#include <codecvt>
#include <string>
#include <regex>

// #include "ctc_word_beam_search/DataLoader.hpp"
// #include "ctc_word_beam_search/WordBeamSearch.hpp"
// #include "ctc_word_beam_search/Metrics.hpp"

#include "SymspellCPP/include/SymSpell.h"

const int initialCapacity = 82765;
const int maxEditDistance = 2;
const int prefixLength = 3;
SymSpell sym_spell = SymSpell(initialCapacity, maxEditDistance, prefixLength);

using namespace std;
enum charTypeT{ other, alpha, digit};

charTypeT charType(char c){
    if(isdigit(c))return digit;
    if(isalpha(c))return alpha;
    return other;
}

string separateThem(string inString){
  string oString = "";charTypeT st=other;
    for(auto c:inString){
        if( (st==alpha && charType(c)==digit) || (st==digit && charType(c)==alpha) )
          oString.push_back(' ');
        oString.push_back(c);st=charType(c);
    }
    return oString;
}

#define OUTPUT_LAYER_DIMS 2048

static bool configLoaded = false;

static const std::wstring numbers = L"0123456789";
static const std::wstring symbols = L"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
static const std::vector<std::string> vocabulary = { 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=",">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", " ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" }; //, "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Í", "Î", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "Ø", "Ú", "Û", "Ü", "Ý", "Þ", "ß", "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ", "ą", "ę", "Į", "į", "ı", "Ł", "ł", "Œ", "œ", "Š", "š", "ų", "Ž", "ž"};
static std::wstring recognizable_chars;
static std::wstring ignore_chars;
static std::vector<unsigned int> ignore_idx;
static std::vector<std::string> lang_dict;
static bool rtl = false;

// CTC decoder parameters
// const LanguageModelType lm_type =
//   LanguageModelType::Words;
// std::shared_ptr<LanguageModel> lm;
// std::shared_ptr<Metrics> metrics;

// symspell checker
auto str_with_numbers = std::regex("^(?=[^\s]*?[0-9])(?=[^\s]*?[a-zA-Z])[a-zA-Z0-9]*$");

extern "C" std::string postProcessText(
  std::string *text)
{
    std::vector<std::string> splitted_text;
    boost::algorithm::split(splitted_text, *text, boost::is_any_of(" "));

    std::cmatch cm;
    std::string new_text;
    for (const auto& t: splitted_text) {
        std::regex_match (t.c_str(), cm, str_with_numbers, std::regex_constants::match_default);
      if (cm.empty()) {
        new_text += t;
        new_text += " ";
      } else {
        for (const auto matches: cm) {
          auto letters_and_numbers = separateThem(matches);
          new_text += letters_and_numbers;
          new_text += " ";
        }
      }
    }
    std::transform(new_text.begin(), new_text.end(), new_text.begin(),
        [](unsigned char c){ return std::tolower(c); });

    vector<xstring> sentences = {  XL(new_text.c_str()) };
    auto ignore_regex = std::regex("[0-9]+");

    auto start_t = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < sentences.size(); i++)
    {
        vector<SuggestItem> results = sym_spell.LookupCompound(sentences[i], 2, ignore_regex);
        xcout << sentences[i] << XL(" -> ") << results[0].term << endl;
    }
    auto end_t = std::chrono::high_resolution_clock::now();
    std::cout <<"Time taken: "<< std::chrono::duration<double>(end_t - start_t).count() << std::endl;
    return "";
}

extern "C" bool loadSymSpellDictionary(
  const char *app_root,
  const std::string& model_name = "latin",
  const std::string& lang = "en")
{
  const std::string dict_dir =
    std::string(app_root) +
    "src/nvdsinfer_parsers/text_recognition/dict/" +
    lang + ".txt";
  // std::cout << "dict_dir:" << dict_dir << std::endl;
  sym_spell.LoadDictionary(dict_dir, 0, 1, XL(' '));
  return true;
}

// extern "C" bool loadCTCModels(
//   const char *app_root,
//   const std::string& model_name = "latin",
//   const std::string& lang = "en")
// {
//   const std::string baseDir =
//     std::string(app_root) +
//     "src/nvdsinfer_parsers/text_recognition/language_configs/" +
//     model_name;
//   std::cout << "baseDir :" << baseDir << std::endl;
// 	const size_t sampleEach = 1; // only take each k*sampleEach sample from dataset, with k=0, 1, ...
// 	const double addK = 1.0; // add-k smoothing of bigram distribution
// 	DataLoader loader{ baseDir, sampleEach, lm_type, addK }; // load data
// 	lm = loader.getLanguageModel(); // get LM
// 	metrics = std::shared_ptr<Metrics>(new Metrics({ lm->getWordChars()})); // CER and WER
//   return true;
// }

extern "C" bool loadModelChars(
    const char *app_root,
    const std::string& model_name = "latin",
    const std::string& lang = "en")
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
        // std::cout << "recognizable_chars:" << std::string(converter.to_bytes(recognizable_chars)).c_str() << std::endl;

        rtl = model_configs[model_name]["rtl"].asBool();
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

extern "C" bool loadLanguageChars(const char *app_root, const std::string& lang = "en")
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
        ignore_idx.clear();
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

        for (unsigned int i = 0; i < recognizable_chars.size(); ++i) 
        {
            bool remove = true;
            const auto &rc = recognizable_chars[i];
            for (const auto &lc : lang_chars)
            {
                if (rc == lc)
                {
                    remove = false;
                }
            }

            if (remove)
            {
                ignore_idx.push_back(i);
                // std::cout << "ignore_idx:" << i << std::endl;
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

extern "C" bool loadLanguageDict(const char *app_root, const std::string& lang = "en")
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
            !loadLanguageDict(app_root) ||
            !loadSymSpellDictionary(app_root)) 
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
            // std::vector<float> maxProb(dims.c, 0.0);
            // std::vector<int> maxProbIdx(dims.c, -1);
            // for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
            // {
            //     float prob = 0.0;
            //     for (unsigned int hIdx = 0; hIdx < dims.h; hIdx++)
            //     {
            //         prob = outputEmbedding[cIdx * dims.h + hIdx];
            //         if (prob > maxProb[cIdx])
            //         {
            //             maxProb[cIdx] = prob;
            //             maxProbIdx[cIdx] = hIdx;
            //         }
            //     }
            // }
            const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            std::vector<std::vector<double>> mat_data;
            for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
            {
                mat_data.push_back(std::vector<double>());
                for (unsigned int hIdx = 1; hIdx < dims.h; hIdx++)
                {
                    if (std::find(ignore_idx.begin(), ignore_idx.end(), hIdx) == ignore_idx.end()) {
                        mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h + hIdx]);
                    }
                }
                mat_data[cIdx].push_back(outputEmbedding[cIdx * dims.h]);
            }
            std::vector<std::pair<double, std::string>> results = ctc_beam_search_decoder(mat_data, vocabulary, 5);
            // std::string results = ctc_greedy_decoder(mat_data, vocabulary);
            std::string output = results[0].second;
            const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
            std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime-startTime).count() << "ms\n\n";
            // // decode it
            // MatrixCustom mat(mat_data);
            // const std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            // const auto res = wordBeamSearch(mat, 25, lm, lm_type);
            // std::cout << "Result:       \"" << lm->labelToUtf8(res) << "\"\n";
            // const std::chrono::system_clock::time_point currTime = std::chrono::system_clock::now();
            // std::cout << "Average Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(currTime-startTime).count() << "ms\n\n";
            // std::wstring output;
            // std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
            // for (unsigned int cIdx = 0; cIdx < dims.c; ++cIdx)
            // {
            //     const auto &charIdx = maxProbIdx[cIdx];
            //     if (charIdx > 0 && !(cIdx > 0 && charIdx == maxProbIdx[cIdx - 1]))
            //     {
            //         if (
            //             ignore_chars.find(recognizable_chars[charIdx - 1]) ==
            //             std::wstring::npos)
            //         {
            //             output += recognizable_chars[charIdx - 1];
            //         }
            //     }
            // }

            if (!output.empty())
            {
                // auto label = new std::string(converter.to_bytes(output));
                auto label = new std::string(output);
                postProcessText(label);
                std::cout << label->c_str() << std::endl;
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

        if (!loadModelChars(app_root, "arabic", "ur") ||
            !loadLanguageChars(app_root, "ur") ||
            !loadLanguageDict(app_root, "ur"))
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
                    // if (
                    //     ignore_chars.find(recognizable_chars[charIdx - 1]) ==
                    //     std::wstring::npos)
                    // {
                    output += recognizable_chars[charIdx - 1];
                    // }
                }
            }

            if (!output.empty())
            {
                if (rtl) {
                    std::reverse(output.begin(), output.end());
                }
                auto label = new std::string(converter.to_bytes(output));
                // std::cout << "Post processed text:" << postProcessText(label) << std::endl;
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
            << "NvDsInferTextRecognitionArabicParser: "
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