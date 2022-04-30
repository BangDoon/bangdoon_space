package com.dsc.oasis.users.utils;

public interface Encrypt {
    String hashPassword(String password);
    boolean comparePassword(String password, String hashed);
    String makeJwt(Long userId) throws Exception;
    boolean checkJwt(String jwt) throws Exception;
}
