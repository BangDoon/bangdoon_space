package com.dsc.oasis.trashcan.application;

import com.dsc.oasis.trashcan.domain.TrashCan;
import com.dsc.oasis.trashcan.domain.TrashCanGeometryRepository;
import com.dsc.oasis.trashcan.domain.TrashCanRepository;
import com.dsc.oasis.trashcan.dto.TrashCanResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@RequiredArgsConstructor
@Service
public class TrashcanService {

    private final TrashCanRepository tCanRepository;

    private final TrashCanGeometryRepository tCanGeoRepository;

    public List<TrashCanResponse> getTrashCans() {

        List<TrashCanResponse> TrashCanResponses=tCanRepository.findAll()
                .stream()
                .map(TrashCan::toResponse)
                .collect(Collectors.toList());

        return TrashCanResponses;
    }

    public TrashCanResponse getTrashCanById(Long id) {

        TrashCanResponse TrashCanResponse=tCanRepository.findById(id).orElse(null)
                .toResponse();

        return TrashCanResponse;
    }

    public List<TrashCanResponse> getNearTrashCans(Double latitude,Double longitude){

        List<TrashCanResponse> TrashCanResponses=tCanGeoRepository.getNearTCans(latitude,longitude)
                .stream()
                .map(TrashCan::toResponse)
                .collect(Collectors.toList());

        return TrashCanResponses;
    }

//    public void createTCan() throws ParseException {
//        TrashCanResponse<List>
//
//        for(Double latitude, Double longitude)
//        String pointWKT=String.format("POINT(%s %s)",longitude,latitude);
//
//        Point point=(Point)new WKTReader().read(pointWKT);
//        TrashCanResponse tCan= TrashCanResponse.builder().point(point).build();
//        tCanRepository.save(tCan);
//    }
}
