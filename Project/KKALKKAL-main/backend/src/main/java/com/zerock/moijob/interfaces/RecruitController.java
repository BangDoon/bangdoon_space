package com.zerock.moijob.interfaces;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.zerock.moijob.application.RecruitService;
import com.zerock.moijob.dto.RecruitRequest;
import com.zerock.moijob.dto.RecruitResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.io.IOException;
import java.util.List;

@RequiredArgsConstructor
@RestController
public class RecruitController {

    private final RecruitService recruitService;


    @GetMapping("/recruits")
    public List<RecruitResponse> detail(
            @RequestParam(value="region",required = false, defaultValue = "") String region
    ){
        List<RecruitResponse> recruitResponses= recruitService.getRecruits(region);

        return recruitResponses;
    }

    @Scheduled(cron = "0 0 12,15,18 * * *", zone="Asia/Seoul")
//    @PostMapping("/recruits")
    public ResponseEntity<?> update() throws IOException {
        ObjectMapper objectMapper=new ObjectMapper();

        String filePath="C:\\Users\\admin\\data.json";

        List<RecruitRequest> recruitRequests=objectMapper.readValue(new File(filePath),
                new TypeReference<List<RecruitRequest>>() {});

        recruitService.bulkUpdate(recruitRequests);

        return ResponseEntity.ok("Update All Success");
    }

}
