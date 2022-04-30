package com.dsc.oasis.users.user;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<OasisUser, Long> {

    @Query("select s from OasisUser s where s.email=?1")
    Optional<OasisUser> findUserByEmail(String email);

    @Query("select s from OasisUser s where s.id=?1")
    Optional<OasisUser> findUserById(Long id);
}
