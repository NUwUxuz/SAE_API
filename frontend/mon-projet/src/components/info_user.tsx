import { useState } from "react";

interface InfoUserProps {
    email: string;
    id: string;
}

export default function InfoUser({ email, id }: InfoUserProps) {
    const [isEditing, setIsEditing] = useState(false);
    const [showPass, setShowPass] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);
    
    const [newEmail, setNewEmail] = useState("");
    const [currentPassword, setCurrentPassword] = useState("");
    const [newPass, setNewPass] = useState("");
    const [confirmPass, setConfirmPass] = useState("");
    const [statusMsg, setStatusMsg] = useState({ type: "", txt: "" });

    const isLengthValid = newPass === "" || newPass.length >= 13;
    const isMatching = newPass === confirmPass;
    const canSubmit = currentPassword !== "" && isLengthValid && isMatching;

    const handleUpdate = async (e: React.FormEvent) => {
        e.preventDefault();
        const token = localStorage.getItem("token");
        const updateData: any = { current_password: currentPassword };
        if (newEmail.trim() !== "") updateData.email = newEmail.trim().toLowerCase();
        if (newPass) updateData.new_mdp = newPass;

        try {
            const response = await fetch(`http://127.0.0.1:8000/user/${id}`, {
                method: "PATCH",
                headers: {
                    "Authorization": "Bearer " + token,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(updateData)
            });
            const result = await response.json();
            if (response.ok) {
                setStatusMsg({ type: "success", txt: "Profil mis à jour !" });
                setTimeout(() => window.location.reload(), 1500);
            } else {
                setStatusMsg({ type: "error", txt: result.detail || "Erreur" });
            }
        } catch (err) {
            setStatusMsg({ type: "error", txt: "Erreur serveur" });
        }
    };

    return (
        <div className="user-info-sidebar">
            <div className="info-group">
                <label>E-mail actuel</label>
                <p className="static-text">{email}</p>
            </div>

            <button className={`btn-edit-profile ${isEditing ? "active" : ""}`} onClick={() => setIsEditing(!isEditing)}>
                {isEditing ? "Annuler" : "Modifier mes informations"}
            </button>

            {isEditing && (
                <form className="edit-mini-form transition-fade" onSubmit={handleUpdate}>
                    {statusMsg.txt && <div className={`status-badge ${statusMsg.type}`}>{statusMsg.txt}</div>}
                    
                    <div className="input-wrapper">
                        <label className="mini-label">Nouvel Email</label>
                        <input type="email" placeholder={email} className="sidebar-input" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} />
                    </div>

                    <div className="input-wrapper" style={{ position: 'relative' }}>
                        <div className="label-with-info">
                            <label className="mini-label">Nouveau mot de passe</label>
                            <div className="info-tooltip-container">
                                <span className="info-icon">i</span>
                                <span className="tooltip-text">Le mot de passe doit contenir au moins 13 caractères.</span>
                            </div>
                        </div>
                        <input 
                            type={showPass ? "text" : "password"} 
                            value={newPass}
                            onChange={(e) => setNewPass(e.target.value)}
                            className={`sidebar-input ${!isLengthValid ? "input-error" : ""}`} 
                        />
                        <button type="button" className="eye-toggle" onClick={() => setShowPass(!showPass)}>
                            {showPass ? "🔒" : "👁️"}
                        </button>
                    </div>

                    <div className="input-wrapper" style={{ position: 'relative' }}>
                        <label className="mini-label">Confirmer nouveau mot de passe</label>
                        <input 
                            type={showConfirm ? "text" : "password"} 
                            value={confirmPass}
                            onChange={(e) => setConfirmPass(e.target.value)}
                            className={`sidebar-input ${!isMatching ? "input-error" : ""}`} 
                        />
                        <button type="button" className="eye-toggle" onClick={() => setShowConfirm(!showConfirm)}>
                            {showConfirm ? "🔒" : "👁️"}
                        </button>
                    </div>

                    <div className="input-wrapper" style={{ borderTop: "1px solid var(--color-border)", paddingTop: "10px" }}>
                        <label className="mini-label" style={{ color: "var(--color-primary)" }}>Mot de passe actuel (Requis)</label>
                        <input type="password" required className="sidebar-input" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} />
                    </div>

                    <button type="submit" className="btn-save" disabled={!canSubmit}>Enregistrer</button>
                </form>
            )}
        </div>
    );
}