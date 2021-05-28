#include "text_post_processor.h"

std::map<std::string, std::shared_ptr<SymSpell>>
    TextPostProcessor::spell_checker;
std::regex TextPostProcessor::str_with_numbers_regex =
    std::regex("^(?=[^\s]*?[0-9])(?=[^\s]*?[a-zA-Z])[a-zA-Z0-9]*$");
std::regex TextPostProcessor::numbers_regex =
    std::regex("[0-9]+");