#pragma once

#include <algorithm>
#include <boost/algorithm/string.hpp>
#ifdef DEBUG
#include <chrono>
#endif
// #include <cstring>
// #include <fstream>
// #include <fcntl.h>
// #include <iostream>
// #include <jsoncpp/json/json.h>
// #include <math.h>
// #include <set>
// #include <vector>
// #include "nvdsinfer_custom_impl.h"

// #include <clocale>
// #include <locale>
// #include <codecvt>
// #include <string>
// #include <regex>

#include "sym_spell_cpp/include/helpers.h"
#include "sym_spell_cpp/include/sym_spell.h"

enum CharacterType
{
    other,
    alpha,
    digit
};

class TextPostProcessor
{
public:
    TextPostProcessor() = default;
    ~TextPostProcessor() = default;

    static CharacterType getCharType(const char &c)
    {
        if (std::isdigit(c))
            return CharacterType::digit;
        if (std::isalpha(c))
            return CharacterType::alpha;
        return CharacterType::other;
    }

    static std::string sepTextAndNumbers(const std::string &input_str)
    {
        std::string output_str = "";
        CharacterType char_type = CharacterType::other;
        for (const auto &ch : input_str)
        {
            if (
                (char_type == CharacterType::alpha &&
                 getCharType(ch) == CharacterType::digit) ||
                (char_type == digit &&
                 getCharType(ch) == CharacterType::alpha))
            {
                output_str.push_back(' ');
            }
            output_str.push_back(ch);
            char_type = getCharType(ch);
        }
        return output_str;
    }

    static bool loadSpellCheckerDictionary(
        const char *app_root,
        const std::string &lang = "en")
    {
        static bool dictionary_loaded = false;
        if (!dictionary_loaded)
        {
            std::cout
                << "Loading dictionary for language: "
                << lang
                << std::endl;
            const std::string dict_dir =
                std::string(app_root) +
                "src/nvdsinfer_parsers/text_recognition/dict/" +
                lang + ".txt";
            auto sc = std::shared_ptr<SymSpell>(
                new SymSpell(
                    TextPostProcessor::initial_capacity,
                    TextPostProcessor::max_edit_distance,
                    TextPostProcessor::prefix_length));
            spell_checker.insert(
                std::pair<std::string, std::shared_ptr<SymSpell>>(lang, sc));
            if (
                !spell_checker[lang]->LoadDictionary(dict_dir, 0, 1, XL(' ')))
            {
                return false;
            }
        }
        return true;
    }

    static xstring processText(
        const std::string &text,
        const std::string &lang = "en")
    {
        std::vector<std::string> splitted_text;
        boost::algorithm::split(splitted_text, text, boost::is_any_of(" "));

        // separate text and numbers
        std::cmatch cm;
        std::string processed_text;
        for (const auto &t : splitted_text)
        {
            std::regex_match(
                t.c_str(),
                cm,
                str_with_numbers_regex,
                std::regex_constants::match_default);
            if (cm.empty())
            {
                processed_text += t;
                processed_text += " ";
            }
            else
            {
                for (const auto matches : cm)
                {
                    auto letters_and_numbers = sepTextAndNumbers(matches);
                    processed_text += letters_and_numbers;
                    processed_text += " ";
                }
            }
        }

        // convert text to lower case
        std::transform(
            processed_text.begin(), processed_text.end(), processed_text.begin(),
            [](unsigned char c) { return std::tolower(c); });

        // make a xstring from processed text
        std::vector<xstring> sentences = {XL(processed_text.c_str())};

#ifdef DEBUG
        // start time
        auto start_t = std::chrono::high_resolution_clock::now();
#endif
        // perform lookup on the sentence
        vector<SuggestItem> results =
            spell_checker[lang]->LookupCompound(
                sentences.back(), max_edit_distance, numbers_regex);
#ifdef DEBUG
        // end ti
        me auto end_t = std::chrono::high_resolution_clock::now();
        std::cout
            << "TextPostProcessor: Time taken by dictionary loookup: "
            << std::chrono::duration<double>(end_t - start_t).count()
            << std::endl;
#endif
        return results[0].term;
    }

private:
    static std::regex str_with_numbers_regex; // regex for pairs of letters and numbers
    static std::regex numbers_regex;          // regex for numbers
    static std::map<std::string, std::shared_ptr<SymSpell>> spell_checker;
    static const int initial_capacity = 82765;
    static const int prefix_length = 3;
    static const int max_edit_distance = 1;
};