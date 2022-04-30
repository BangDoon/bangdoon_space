package com.dsc.oasis.trashcan.domain;

import com.dsc.oasis.trashcan.dto.TrashCanResponse;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.*;
import net.minidev.json.annotate.JsonIgnore;
import org.locationtech.jts.geom.Point;

import javax.persistence.*;
import javax.persistence.criteria.CriteriaBuilder;

@Entity
@Builder
@Getter
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Table(name= "tb1_tcan")
public class TrashCan {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String address;

    private String location;

    private String trash_type;

    private Double latitude;

    private Double longitude;

    private Integer distance;

    private Point point;

    private Point point_reverse;

    public TrashCanResponse toResponse(){
        return TrashCanResponse.builder()
                .id(id)
                .address(address)
                .location(location)
                .trash_type(trash_type)
                .latitude(latitude)
                .longitude(longitude)
                .distance(distance).build();
    }
}
