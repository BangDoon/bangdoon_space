package com.zerock.moijob.application;

import com.zerock.moijob.domain.Recruit;
import com.zerock.moijob.domain.RecruitRepository;
import com.zerock.moijob.dto.RecruitRequest;
import com.zerock.moijob.dto.RecruitResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@RequiredArgsConstructor
@Service
public class RecruitService {

    private final RecruitRepository recruitRepository;


    public List<RecruitResponse> getRecruits(String region) {

        List<RecruitResponse> recruitResponses=recruitRepository.findAllByRegionContaining(region).stream()
                .map(Recruit::toResponse)
                .collect(Collectors.toList());;

        return recruitResponses;
    }

    public void bulkUpdate(List<RecruitRequest> requestList){
        recruitRepository.deleteAll();

        for(RecruitRequest request : requestList)
        {
            recruitRepository.save(request.toEntity());
        }
    }
}
