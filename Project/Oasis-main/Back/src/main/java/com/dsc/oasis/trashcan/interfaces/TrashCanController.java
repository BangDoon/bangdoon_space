package com.dsc.oasis.trashcan.interfaces;

import com.dsc.oasis.trashcan.application.TrashcanService;
import com.dsc.oasis.trashcan.domain.TrashCan;
import com.dsc.oasis.trashcan.dto.TrashCanResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RequiredArgsConstructor
@RestController
@RequestMapping(path = "api/oasis")
public class TrashCanController {

    private final TrashcanService trashcanService;

    @GetMapping("/trashcans/{id}")
    public TrashCanResponse listById(
            @PathVariable("id") Long id
    ){
        TrashCanResponse trashCanResponses=trashcanService.getTrashCanById(id);

        return trashCanResponses;
    }

    @GetMapping("/trashcans")
    public List<TrashCanResponse> listByNear(
            @RequestParam("lat") Double lat,
            @RequestParam("lon") Double lon
    ){
        List<TrashCanResponse> trashCanResponses =trashcanService.getNearTrashCans(lat,lon);

        return trashCanResponses;
    }


}
