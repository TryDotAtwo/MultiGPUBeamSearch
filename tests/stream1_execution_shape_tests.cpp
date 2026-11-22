#include "../cuda/stream1_execution_shape.hpp"
#include <sstream>
#include <string>
int main() {
    char uuid[16]{};
    std::ostringstream invalid_uuid;
    try { beam::write_stream1_device_identity(invalid_uuid,0,86,uuid); return 6; }
    catch(const std::invalid_argument&) { if(!invalid_uuid.str().empty()) return 7; }
    uuid[0]=1; uuid[15]=static_cast<char>(255);
    std::ostringstream identity;
    beam::write_stream1_device_identity(identity,0,86,uuid);
    if(identity.str().find("010000000000000000000000000000ff") == std::string::npos) return 8;
    std::ostringstream valid;
    beam::write_stream1_execution_shape(valid,1024,256,4,1,90);
    if (valid.str().find("\"transformer_microbatch\":256") == std::string::npos ||
        valid.str().find("\"production_quality_accepted\":false") == std::string::npos) return 1;
    const unsigned shapes[][3]={{0,1,1},{1,0,1},{1,2,1},{1,1,0}};
    for (const auto& s:shapes) {
        std::ostringstream rejected;
        try { beam::write_stream1_execution_shape(rejected,s[0],s[1],s[2],0,86); return 2; }
        catch(const std::invalid_argument&) { if(!rejected.str().empty()) return 3; }
    }
    for(const auto& pair : {std::pair<int,int>{-1,86},{0,74}}) {
        std::ostringstream rejected;
        try { beam::write_stream1_execution_shape(rejected,8,8,1,pair.first,pair.second); return 4; }
        catch(const std::invalid_argument&) { if(!rejected.str().empty()) return 5; }
    }
    return 0;
}
