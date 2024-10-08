#include <ranges>
#include <string>
#include <deque>
#include <iostream>
#include <string_view>

const std::string EMPTY = "";
const std::string UP = "..";
const std::string CUR = ".";

class Solution {
public:
    std::string simplifyPath(std::string path) {
        std::deque<std::string> out_tokens = {"/"};
        for(const auto token : std::views::split(path, '/')) {
            auto tv = std::string_view(token.begin(), token.end());
            std::cout << "Token: " << tv << std::endl;
            std::cout << "Tokens size: " << out_tokens.size() << std::endl;

            if(tv == EMPTY) {
                ;
            }
            else if(tv == CUR) {
                ;
            }
            else if(tv == UP) {
                if(out_tokens.size() > 1)
                  out_tokens.pop_back();
                if(out_tokens.size() > 1)
                  out_tokens.pop_back();
            }
            else {
                out_tokens.push_back("/");
                out_tokens.emplace_back(token.begin(), token.end());
            }
        }
        auto j = std::views::join(out_tokens);
        return std::string(j.begin(), j.end());
    }
};

int main() {
    Solution s;
    std::string path = "/../";
    std::string result = s.simplifyPath(path);

    std::cout << "Path: " << path << std::endl;
    std::cout << "Result: " << result << std::endl;

    return 0;
}